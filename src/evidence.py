"""Deterministic evidence extraction, grounding, and confidence checks.

Everything in this module is pure Python — no LLM calls — so results are
reproducible and unit-testable. The LLM proposes; this module verifies.
"""
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from src.schemas import EvidenceItem, FailureAnalysis


@dataclass
class ErrorEvent:
    """A single suspicious log line with its position and severity."""

    line_number: int  # 1-based, as it appears in the pasted log
    text: str
    kind: str         # pattern family that matched
    weight: float     # severity weight used to rank root-cause candidates


# Pattern families ordered by how strongly they indicate a root cause.
# A line is classified by the first family that matches, so specific
# families (traceback, exception) must precede the generic 'error'.
ERROR_PATTERNS: list[tuple[str, "re.Pattern[str]", float]] = [
    ("traceback", re.compile(r"(?i)\btraceback\b"), 1.0),
    (
        "exception",
        re.compile(
            r"(?i)\bexception\b|assertionerror|modulenotfounderror|"
            r"importerror|cannot find module"
        ),
        0.95,
    ),
    (
        # CamelCase exception names — case-sensitive on purpose so plain
        # 'error' text still classifies as the generic 'error' kind.
        "exception",
        re.compile(r"\b\w*Error\b|\b\w*Exception\b"),
        0.95,
    ),
    ("fatal", re.compile(r"(?i)\bfatal\b|\bpanic\b"), 0.9),
    ("failed", re.compile(r"(?i)\bfailed\b|\bfailure\b|command failed"), 0.8),
    ("error", re.compile(r"(?i)(?:\berror\b|\berr!)"), 0.7),
    ("exit_code", re.compile(r"(?i)exit code [1-9]"), 0.5),
]

MAX_EVENTS = 20          # bound prompt size on very noisy logs
_OVERLAP_THRESHOLD = 0.6  # min token overlap for a fuzzy evidence match


def match_error(line: str) -> tuple[str, float] | None:
    """Return (kind, weight) of the highest-priority pattern matching a line."""
    for kind, pattern, weight in ERROR_PATTERNS:
        if pattern.search(line):
            return kind, weight
    return None


def extract_error_events(error_log: str) -> list[ErrorEvent]:
    """Pull every suspicious log line out, tagged with a 1-based line number."""
    events: list[ErrorEvent] = []
    for i, line in enumerate(error_log.splitlines()):
        text = line.strip()
        if not text:
            continue
        matched = match_error(line)
        if matched:
            kind, weight = matched
            events.append(ErrorEvent(i + 1, text[:300], kind, weight))
        if len(events) >= MAX_EVENTS:
            break
    return events


def rank_error_events(events: list[ErrorEvent]) -> list[ErrorEvent]:
    """Order events so the most likely root cause comes first.

    Higher severity wins; among equal severity the earlier line wins — CI
    jobs abort on the first real failure, so later events are usually
    consequences. Symptoms like 'exit code 1' therefore never outrank a
    genuine error line.
    """
    return sorted(events, key=lambda ev: (-ev.weight, ev.line_number))


# ---------------------------------------------------------------------------
# Evidence grounding: verify LLM claims against the actual log and files.
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _token_overlap(needle: str, haystack: str) -> float:
    """Fraction of distinct needle tokens present in the haystack."""
    tokens = set(_normalize(needle).split())
    if not tokens:
        return 0.0
    return len(tokens & set(_normalize(haystack).split())) / len(tokens)


def _find_log_reference(error_log: str, claim: str) -> tuple[int, str] | None:
    """Locate a claim inside the log; return (line_number, matched_line)."""
    norm_claim = _normalize(claim)
    if not norm_claim:
        return None

    lines = error_log.splitlines()
    for i, line in enumerate(lines):
        norm_line = _normalize(line)
        if not norm_line:
            continue
        if norm_claim in norm_line or (len(norm_line) >= 15 and norm_line in norm_claim):
            return i + 1, line.strip()

    # Fuzzy fallback: keep the best token-overlap match.
    best: tuple[float, int, str] | None = None
    for i, line in enumerate(lines):
        score = _token_overlap(claim, line)
        if score >= _OVERLAP_THRESHOLD and (best is None or score > best[0]):
            best = (score, i, line.strip())
    if best:
        return best[1] + 1, best[2]
    return None


def _file_reference(path: str, line_start: int | None, line_end: int | None) -> str:
    if line_start is None:
        return path
    if line_end is None or line_end == line_start:
        return f"{path}:{line_start}"
    return f"{path}:{line_start}-{line_end}"


# File references embedded inside log lines: Python tracebacks and the
# eslint/gcc style 'path:line(:col)'. The second pattern requires a file
# extension so '12:5  error' columns are not mistaken for paths.
_FILE_REF_PATTERNS = [
    re.compile(r'File "(?P<path>[^"]+)", line (?P<line>\d+)'),
    re.compile(r"(?P<path>(?:[A-Za-z]:)?[\w.\-/\\]+\.\w+):(?P<line>\d+)"),
]


def _extract_file_refs(line: str) -> list[tuple[str, int]]:
    """Pull (path, line_number) references out of a log line."""
    refs: list[tuple[str, int]] = []
    for pattern in _FILE_REF_PATTERNS:
        for m in pattern.finditer(line):
            ref = (m.group("path"), int(m.group("line")))
            if ref not in refs:
                refs.append(ref)
    return refs


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./").lower()


def _doc_for_path(docs: list[Document], ref_path: str) -> Document | None:
    """Find the retrieved doc a referenced path points at.

    A traceback may show an absolute path ('C:\\proj\\src\\json.py') or a
    bare filename ('json.py') while docs carry repo-relative paths, so we
    try normalized suffix matches first and fall back to basename equality.
    Files absent from the retrieved context (e.g. stdlib paths) return None.
    """
    norm_ref = _normalize_path(ref_path)
    basename = norm_ref.rsplit("/", 1)[-1]
    best: Document | None = None
    for doc in docs:
        norm_doc = _normalize_path(doc.metadata.get("path", ""))
        if not norm_doc:
            continue
        if (
            norm_ref == norm_doc
            or norm_ref.endswith("/" + norm_doc)
            or norm_doc.endswith("/" + norm_ref)
        ):
            return doc
        if best is None and norm_doc.rsplit("/", 1)[-1] == basename:
            best = doc
    return best


def _file_item_for_log_line(
    line_no: int, line_text: str, docs: list[Document]
) -> EvidenceItem | None:
    """Upgrade a log line that cites a retrieved repo file to file evidence.

    A traceback like ``File "C:\\proj\\json.py", line 3`` is log evidence
    *and* repository corroboration when json.py was retrieved — the item is
    classified 'file' so reconcile_confidence counts it correctly. Returns
    None when no referenced path resolves to a retrieved doc.
    """
    if not docs:
        return None
    for ref_path, ref_line in _extract_file_refs(line_text):
        doc = _doc_for_path(docs, ref_path)
        if doc is None:
            continue
        path = doc.metadata.get("path", ref_path)
        line_start = doc.metadata.get("line_start")
        line_end = doc.metadata.get("line_end")
        body = doc.page_content.splitlines()
        body = body[1:] if doc.page_content.startswith("# File:") else body
        if line_start is not None and line_end is not None and line_start <= ref_line <= line_end:
            reference = _file_reference(path, ref_line, ref_line)
            excerpt = body[ref_line - line_start].strip()[:300]
        else:
            reference = _file_reference(path, line_start, line_end)
            excerpt = "\n".join(body).strip()[:300]
        return EvidenceItem(
            kind="file",
            reference=f"log line {line_no} → {reference}",
            excerpt=excerpt,
            verified=True,
        )
    return None


def _find_file_reference(
    docs: list[Document], claim: str
) -> tuple[str, int | None, int | None, str] | None:
    """Locate a claim inside retrieved docs.

    Returns (path, line_start, line_end, excerpt). Chunks carry
    ``line_start``/``line_end`` metadata added by the indexer.
    """
    norm_claim = _normalize(claim)
    if not norm_claim:
        return None

    # Pass 1: a single line inside a chunk contains the claim (or vice versa).
    for doc in docs:
        header = 1 if doc.page_content.startswith("# File:") else 0
        chunk_start = doc.metadata.get("line_start")
        for j, line in enumerate(doc.page_content.splitlines()[header:]):
            norm_line = _normalize(line)
            if not norm_line:
                continue
            if norm_claim in norm_line or (len(norm_line) >= 15 and norm_line in norm_claim):
                line_no = chunk_start + j if chunk_start else None
                return (
                    doc.metadata.get("path", "unknown"),
                    line_no,
                    line_no,
                    line.strip()[:300],
                )

    # Pass 2: explicit path mention or best token overlap at chunk level.
    best_score, best_doc = 0.0, None
    for doc in docs:
        path = doc.metadata.get("path", "")
        score = 1.0 if path and path in claim else _token_overlap(claim, doc.page_content)
        if score > best_score:
            best_score, best_doc = score, doc
    if best_doc is not None and best_score >= _OVERLAP_THRESHOLD:
        body = best_doc.page_content.splitlines()
        body = body[1:] if best_doc.page_content.startswith("# File:") else body
        return (
            best_doc.metadata.get("path", "unknown"),
            best_doc.metadata.get("line_start"),
            best_doc.metadata.get("line_end"),
            "\n".join(body).strip()[:300],
        )
    return None


def ground_evidence(
    analysis: FailureAnalysis,
    error_log: str,
    retrieved_docs: list[Document],
) -> list[EvidenceItem]:
    """Verify each model-cited evidence string against real artifacts.

    Every claim becomes an EvidenceItem: matched log lines get a
    ``log line N`` reference, matched file content gets ``path:start-end``,
    and claims that match nothing are kept but flagged ``verified=False``
    so the UI can label them as unverified model output.
    """
    items: list[EvidenceItem] = []
    for raw_claim in analysis.evidence:
        claim = raw_claim.strip()
        if not claim:
            continue

        log_hit = _find_log_reference(error_log, claim)
        if log_hit:
            line_no, line_text = log_hit
            # A log line that cites a retrieved repo file (e.g. a traceback
            # frame) corroborates via the file, not just the log.
            file_item = _file_item_for_log_line(line_no, line_text, retrieved_docs)
            items.append(
                file_item
                or EvidenceItem(
                    kind="log",
                    reference=f"log line {line_no}",
                    excerpt=line_text,
                    verified=True,
                )
            )
            continue

        file_hit = _find_file_reference(retrieved_docs, claim)
        if file_hit:
            path, line_start, line_end, excerpt = file_hit
            items.append(
                EvidenceItem(
                    kind="file",
                    reference=_file_reference(path, line_start, line_end),
                    excerpt=excerpt,
                    verified=True,
                )
            )
            continue

        items.append(
            EvidenceItem(
                kind="log",
                reference="",
                excerpt=claim[:300],
                verified=False,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Confidence reconciliation: cap the model's score by evidence quality.
# ---------------------------------------------------------------------------

def reconcile_confidence(
    analysis: FailureAnalysis,
    evidence_items: list[EvidenceItem],
    retrieved_docs: list[Document],
    error_events: list[ErrorEvent],
) -> tuple[float, str, list[str]]:
    """Return (score, level, notes) after cross-checking evidence quality.

    The model's self-reported confidence is a ceiling, not the final word:
    missing or unverifiable evidence caps it. Notes explain each cap so the
    UI can communicate *why* confidence was limited.
    """
    notes: list[str] = []
    cap = 1.0
    verified = [e for e in evidence_items if e.verified]
    file_evidence = [e for e in verified if e.kind == "file"]

    if not error_events:
        cap = min(cap, 0.3)
        notes.append("no error signatures were detected in the log")
    if not verified:
        cap = min(cap, 0.4)
        notes.append("no cited evidence could be verified against the log or files")
    elif len(verified) == 1:
        cap = min(cap, 0.7)
        notes.append("only one piece of cited evidence could be verified")
    if not file_evidence:
        if retrieved_docs:
            cap = min(cap, 0.75)
            notes.append("no project file corroborates the diagnosis")
        else:
            cap = min(cap, 0.6)
            notes.append("no project context was provided")
    if not analysis.evidence_sufficient:
        cap = min(cap, 0.5)
        notes.append("the model flagged the available evidence as insufficient")

    score = round(min(analysis.confidence, cap), 2)
    level = "low" if score < 0.4 else "medium" if score < 0.7 else "high"
    return score, level, notes
