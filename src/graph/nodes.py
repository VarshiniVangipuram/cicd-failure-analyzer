"""LangGraph nodes: parse_log -> retrieve_context -> analyze_failure."""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.changes import (
    FileChange,
    ScoredChange,
    completeness_note,
    correlate_changes,
)
from src.config import settings
from src.evidence import (
    ErrorEvent,
    extract_error_events,
    ground_evidence,
    match_error,
    rank_error_events,
    reconcile_confidence,
)
from src.graph.state import AnalysisState
from src.llm import get_llm
from src.rag.indexer import retrieve
from src.schemas import FailureAnalysis

# ---------------------------------------------------------------------------
# Node 1: deterministic log parsing — no LLM call needed to spot error lines.
# ---------------------------------------------------------------------------

MAX_SIGNATURES = 15
MAX_LOG_CHARS = 12_000  # keep prompt size bounded


def extract_error_signatures(error_log: str) -> list[str]:
    """Pull the most suspicious lines out of a CI/CD log."""
    signatures: list[str] = []
    seen: set[str] = set()

    lines = error_log.splitlines()
    for i, line in enumerate(lines):
        if match_error(line) is None:
            continue
        # Include the following line too — tracebacks/details often sit there.
        for candidate in lines[i : i + 2]:
            stripped = candidate.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                signatures.append(stripped[:300])
        if len(signatures) >= MAX_SIGNATURES:
            break

    if not signatures:
        # Fallback: take the last N lines — failures usually surface at the end.
        signatures = [l.strip() for l in lines[-MAX_SIGNATURES:] if l.strip()]

    return signatures


def parse_log_node(state: AnalysisState) -> dict:
    signatures = extract_error_signatures(state["error_log"])
    events = rank_error_events(extract_error_events(state["error_log"]))
    links = correlate_changes(
        state.get("changes", []), state["error_log"], events
    )
    return {
        "error_signatures": signatures,
        "error_events": events,
        "primary_error": events[0] if events else None,
        "change_links": links,
    }


# ---------------------------------------------------------------------------
# Node 2: retrieve relevant project files via the vector store.
# ---------------------------------------------------------------------------


def retrieve_context_node(state: AnalysisState) -> dict:
    store = state.get("vectorstore")
    if store is None:
        return {"retrieved_docs": []}

    # Query with each error signature plus a broad failure-type query.
    queries = list(state.get("error_signatures", [])[:8])
    queries.append("CI CD pipeline configuration build test deploy error")

    # Correlated changes are strong retrieval cues — fetch related files.
    for link in state.get("change_links", []):
        if link.score <= 0 or len(queries) >= 14:
            break
        queries.append(f"{link.change.path} {link.change.status}")

    # The user's question steers retrieval toward what they care about.
    question = state.get("user_question", "").strip()
    if question:
        queries.append(question)

    docs = retrieve(store, queries, top_k=settings.top_k)
    return {"retrieved_docs": docs[: settings.top_k * 2]}


def _format_context(docs: list[Document], max_chars: int = 10_000) -> str:
    parts: list[str] = []
    total = 0
    for doc in docs:
        path = doc.metadata.get("path", "unknown")
        line_start = doc.metadata.get("line_start")
        line_end = doc.metadata.get("line_end")
        location = (
            f" (lines {line_start}-{line_end})" if line_start and line_end else ""
        )
        snippet = doc.page_content[:1500]
        block = f"--- {path}{location} ---\n{snippet}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) if parts else "(no project context provided)"


def _format_events(events: list[ErrorEvent], primary: ErrorEvent | None) -> str:
    """Render ranked error events with log line numbers for the prompt."""
    if not events:
        return "(none detected — the log may not contain a clear failure)"
    lines = []
    for i, ev in enumerate(events):
        marker = (
            "   <-- primary suspect"
            if primary is not None and i == 0
            else ""
        )
        lines.append(f"L{ev.line_number} [{ev.kind}] {ev.text}{marker}")
    return "\n".join(lines)


_STATUS_TAG = {"added": "A", "modified": "M", "deleted": "D", "renamed": "R"}


def _format_changes(
    changes: list[FileChange],
    links: list[ScoredChange],
    max_items: int = 8,
) -> str:
    """Render changed files with correlation hints for the prompt."""
    if not changes:
        return "(none provided)"
    by_path = {link.change.path: link for link in links}
    lines = []
    for change in changes[:max_items]:
        header = (
            f"{_STATUS_TAG.get(change.status, '?')} {change.path} "
            f"[{change.category}]"
        )
        if change.patch_incomplete:
            header += " (diff data incomplete)"
        link = by_path.get(change.path)
        if link and link.score > 0:
            header += (
                f" — correlation {link.score:g} ({link.tier}): "
                + "; ".join(link.reasons)
            )
        lines.append(header)
        for line_no, text in change.added_lines[:3]:
            lines.append(f"    +L{line_no}: {text[:100]}")
        for text in change.removed_lines[:2]:
            lines.append(f"    -: {text[:100]}")
    if len(changes) > max_items:
        lines.append(f"… and {len(changes) - max_items} more changed files")
    note = completeness_note(changes)
    if note:
        lines.append(f"NOTE: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 3: LLM analysis with structured output.
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior DevOps engineer and CI/CD debugging expert. "
            "Analyze the failed pipeline log and the retrieved project "
            "context. Identify the failure type and the single most likely "
            "root cause. If the log contains multiple distinct errors, "
            "treat the rest as alternative_causes ordered by likelihood — "
            "do not merge them into one vague explanation. Cite supporting "
            "evidence verbatim (exact log lines or file snippets, including "
            "the shown line numbers when you reference them), list the "
            "relevant files, and give concrete ordered fixes. Assign a "
            "confidence score (0-1), level (low/medium/high), and a "
            "one-sentence confidence_rationale. Set evidence_sufficient to "
            "false when the log and context do not support a reliable "
            "diagnosis — honesty beats confidence. When recent changes are "
            "listed, fill suspect_changes only with changes *likely "
            "related* to the failure — correlation is not causation, so "
            "name the concrete link (cited file, matching error kind, new "
            "identifier in an error line) and leave it empty when nothing "
            "plausibly relates.",
        ),
        (
            "human",
            "## CI/CD Error Log (tail)\n{log_tail}\n\n"
            "## Extracted Error Events (ranked, with log line numbers)\n"
            "{events}\n\n"
            "## Recent Changes\n{changes}\n\n"
            "## Retrieved Project Context\n{context}\n{question_block}\n"
            "Produce the structured failure analysis.",
        ),
    ]
)


def analyze_failure_node(state: AnalysisState) -> dict:
    llm = get_llm()
    structured_llm = llm.with_structured_output(FailureAnalysis)
    chain = ANALYSIS_PROMPT | structured_llm

    retrieved_docs = state.get("retrieved_docs", [])
    error_events = state.get("error_events", [])
    context = _format_context(retrieved_docs)
    events_block = _format_events(error_events, state.get("primary_error"))
    log_tail = state["error_log"][-MAX_LOG_CHARS:]

    # Optional investigation focus — empty keeps the default prompt intact.
    question = state.get("user_question", "").strip()
    question_block = (
        "\n## User's Investigation Request\n"
        f"{question}\n\n"
        "Address this request specifically in your diagnosis and fixes."
        if question
        else ""
    )

    changes_block = _format_changes(
        state.get("changes", []), state.get("change_links", [])
    )
    analysis: FailureAnalysis = chain.invoke(
        {
            "log_tail": log_tail,
            "events": events_block,
            "changes": changes_block,
            "context": context,
            "question_block": question_block,
        }
    )

    # Verify cited evidence and cap confidence deterministically — the
    # model's self-assessment is a ceiling, not the final word.
    evidence_items = ground_evidence(analysis, state["error_log"], retrieved_docs)
    score, level, notes = reconcile_confidence(
        analysis, evidence_items, retrieved_docs, error_events
    )
    analysis.confidence = score
    analysis.confidence_level = level

    return {
        "analysis": analysis,
        "evidence_items": evidence_items,
        "confidence_notes": notes,
    }
