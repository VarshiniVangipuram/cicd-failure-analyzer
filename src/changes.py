"""Change-aware failure analysis: parse diffs, classify, correlate.

Deterministic — no LLM calls. A list of FileChange is produced from pasted
``git diff`` / ``git show`` output, an uploaded .diff/.patch file, or the
GitHub compare API (see src.ingestion.github.load_github_compare).
``correlate_changes`` then scores each change against the failure log so
the most plausible culprits surface first.
"""
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from src.evidence import ErrorEvent


@dataclass
class FileChange:
    """One file touched by a diff."""

    path: str                                        # new (post-change) path
    status: str = "modified"                         # added|modified|deleted|renamed
    category: str = "source"                         # dependency|test|ci|config|docs|source
    old_path: str | None = None                      # set for renames
    added_lines: list[tuple[int, str]] = field(default_factory=list)   # (new line no, text)
    removed_lines: list[str] = field(default_factory=list)
    patch_incomplete: bool = False                   # hunk data missing/truncated


@dataclass
class ScoredChange:
    """A change plus its deterministic correlation score and reasons.

    The tier communicates heuristic strength — 'strong'/'possible'/'weak'
    are *correlation* labels, not proof of causation.
    """

    change: FileChange
    score: float
    reasons: list[str] = field(default_factory=list)
    tier: str = "weak"                               # strong | possible | weak


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_DEPENDENCY_FILES = {
    "requirements.txt", "requirements-dev.txt", "pipfile", "pipfile.lock",
    "poetry.lock", "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.mod", "go.sum", "pom.xml", "build.gradle", "build.gradle.kts",
    "gemfile", "gemfile.lock", "cargo.toml", "cargo.lock",
    "composer.json", "composer.lock", "environment.yml",
}

_CI_FILES = {
    ".gitlab-ci.yml", ".gitlab-ci.yaml", "jenkinsfile", ".travis.yml",
    "azure-pipelines.yml", "azure-pipelines.yaml", "bitbucket-pipelines.yml",
}

_CONFIG_SUFFIXES = (
    ".tf", ".ini", ".cfg", ".toml", ".yml", ".yaml", ".json", ".properties",
)


def classify_change(path: str) -> str:
    """Bucket a changed path so correlation can weigh it correctly."""
    norm = "/" + path.lower().replace("\\", "/")
    name = norm.rsplit("/", 1)[-1]
    if name in _DEPENDENCY_FILES:
        return "dependency"
    if (
        name.startswith("test_")
        or "_test." in name
        or ".test." in name
        or ".spec." in name
        or "/tests/" in norm
        or "/test/" in norm
        or "/__tests__/" in norm
    ):
        return "test"
    if name in _CI_FILES or "/.github/workflows/" in norm or "/.circleci/" in norm:
        return "ci"
    if name in {"dockerfile", "docker-compose.yml", "docker-compose.yaml", ".env", ".env.example"} \
            or name.endswith(_CONFIG_SUFFIXES):
        return "config"
    if name.endswith((".md", ".rst", ".txt")) or "/docs/" in norm:
        return "docs"
    return "source"


# ---------------------------------------------------------------------------
# Unified-diff parsing (git diff / git show / .patch / compare-API patches)
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,\d+)? @@")
_DIFF_GIT_RE = re.compile(r"^diff --git a/(?P<old>.+?) b/(?P<new>.+)$")


def _strip_prefix(path: str) -> str:
    return path[2:] if path.startswith(("a/", "b/")) else path


def apply_patch_text(change: FileChange, patch: str) -> None:
    """Consume ``@@`` hunks into a FileChange (added/removed lines).

    Works on a bare patch body too — used both by the unified-diff parser
    and by the GitHub compare API, which returns a patch per file.
    """
    new_ln = 0
    in_hunk = False
    for line in patch.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            new_ln = int(m.group("start"))
            in_hunk = True
            continue
        if not in_hunk or line.startswith("\\"):
            continue
        if line.startswith("+"):
            change.added_lines.append((new_ln, line[1:]))
            new_ln += 1
        elif line.startswith("-"):
            change.removed_lines.append(line[1:])
        else:  # context line (' ' prefix or bare)
            new_ln += 1


def parse_unified_diff(text: str) -> list[FileChange]:
    """Parse ``git diff``/``git show``/``.patch`` text into FileChanges."""
    changes: list[FileChange] = []
    cur: FileChange | None = None
    new_ln = 0
    in_hunk = False

    def finish() -> None:
        if cur is not None:
            cur.category = classify_change(cur.path or cur.old_path or "")

    for line in text.splitlines():
        m = _DIFF_GIT_RE.match(line)
        if m:
            finish()
            cur = FileChange(
                path=_strip_prefix(m.group("new")),
                old_path=_strip_prefix(m.group("old")),
            )
            changes.append(cur)
            in_hunk = False
            continue
        if cur is None:
            # Headerless unified diff (diff -u style): starts at '--- '.
            if line.startswith("--- "):
                cur = FileChange(path="")
                changes.append(cur)
            else:
                continue  # git show commit header, log text, etc.
        if line.startswith("new file mode"):
            cur.status = "added"
        elif line.startswith("deleted file mode"):
            cur.status = "deleted"
        elif line.startswith("rename from "):
            cur.old_path = line[len("rename from "):].strip()
            cur.status = "renamed"
        elif line.startswith("rename to "):
            cur.path = line[len("rename to "):].strip()
            cur.status = "renamed"
        elif line.startswith("--- "):
            old = line[4:].strip()
            if in_hunk:
                # Headerless diff: hunk ended, a new file's header begins.
                finish()
                cur = FileChange(path="")
                changes.append(cur)
            if old == "/dev/null":
                cur.status = "added"
            else:
                cur.old_path = _strip_prefix(old)
            in_hunk = False
        elif line.startswith("+++ "):
            new = line[4:].strip()
            if new == "/dev/null":
                cur.status = "deleted"
            else:
                cur.path = _strip_prefix(new)
            in_hunk = False
        else:
            hunk = _HUNK_RE.match(line)
            if hunk:
                new_ln = int(hunk.group("start"))
                in_hunk = True
            elif not in_hunk or line.startswith("\\"):
                continue
            elif line.startswith("+"):
                cur.added_lines.append((new_ln, line[1:]))
                new_ln += 1
            elif line.startswith("-"):
                cur.removed_lines.append(line[1:])
            elif line.startswith(" ") or line == "":
                new_ln += 1          # context line
            else:
                in_hunk = False      # index/metadata lines end the hunk
    finish()
    return [c for c in changes if c.path]


# ---------------------------------------------------------------------------
# Correlation: which changes are *likely related* to this failure?
# ---------------------------------------------------------------------------
# Correlation is not causation: scores rank suspects, tiers communicate how
# strong the heuristic link is, and the UI/LLM are told to treat every hit
# as a hypothesis, not a proven cause.

_DEP_LOG_MARKERS = re.compile(
    r"(?i)eresolve|peer dep|no matching distribution|modulenotfounderror|"
    r"importerror|cannot find module|could not find a version|unmet peer|"
    r"missing script|pip.*error"
)
_TEST_LOG_MARKERS = re.compile(
    r"(?i)\bpytest\b|\bunittest\b|\bjest\b|\brspec\b|test.*failed|"
    r"failed.*test|assertionerror|describe\(|expect\("
)
_CI_LOG_MARKERS = re.compile(
    r"(?i)workflow|pipeline|invalid.*ya?ml|syntax error.*config|"
    r"unknown.*step|action.*not found|invalid.*input"
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{3,}")
_STOPWORDS = {
    "import", "from", "return", "self", "this", "const", "function", "class",
    "with", "def", "else", "elif", "await", "async", "print", "then", "void",
    "public", "private", "static", "true", "false", "none", "null", "echo",
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _identifier_hit(change: FileChange, event_text: str) -> str | None:
    """First newly-added identifier that also appears in an error line.

    Dotted names ('self.timeout', 'json.decoder') are tried whole first,
    then by their last component — error lines often cite just the member.
    """
    if not event_text:
        return None
    seen: set[str] = set()
    for _, text in change.added_lines:
        for token in _IDENT_RE.findall(text):
            candidates = [token]
            if "." in token:
                candidates.append(token.rsplit(".", 1)[-1])
            for cand in candidates:
                low = cand.lower()
                if len(low) < 4 or low in _STOPWORDS or low in seen:
                    continue
                seen.add(low)
                if low in event_text:
                    return cand
    return None


def correlation_tier(score: float) -> str:
    """Bucket a correlation score — labels deliberately stop short of 'cause'."""
    if score >= 4:
        return "strong"
    if score >= 2:
        return "possible"
    return "weak"


def correlate_changes(
    changes: list[FileChange],
    error_log: str,
    events: list[ErrorEvent],
) -> list[ScoredChange]:
    """Score each change's plausibility of being related to the failure.

    Signals, strongest first: the file is cited by an error line (4); named
    anywhere in the log (3); identifiers it added appear in error lines (2);
    its category matches the log's failure flavour (dependency manifest +
    dependency errors, test file + test failures, CI file + pipeline errors).
    Returns every change sorted by descending score.
    """
    norm_log = _norm(error_log)
    event_text = " ".join(_norm(e.text) for e in events)

    scored: list[ScoredChange] = []
    for change in changes:
        score, reasons = 0.0, []
        variants = {
            _norm(change.path),
            _norm(change.path.rsplit("/", 1)[-1]),
        }
        in_event = any(v and v in event_text for v in variants)
        in_log = in_event or any(v and v in norm_log for v in variants)

        if in_event:
            score += 4
            reasons.append("file is cited by an error line")
        elif in_log:
            score += 3
            reasons.append("file is named in the log")
        if change.status == "deleted" and in_log:
            score += 1
            reasons.append("file was deleted but is still referenced")

        if change.category == "dependency" and _DEP_LOG_MARKERS.search(error_log):
            score += 2
            reasons.append("dependency manifest changed and the log shows dependency errors")
        elif change.category == "test" and _TEST_LOG_MARKERS.search(error_log):
            score += 2
            reasons.append("test file changed and the failure looks test-related")
        elif change.category == "ci" and _CI_LOG_MARKERS.search(error_log):
            score += 1.5
            reasons.append("CI configuration changed alongside pipeline errors")

        hit = _identifier_hit(change, event_text)
        if hit:
            score += 2
            reasons.append(f"newly added '{hit}' appears in an error line")

        scored.append(
            ScoredChange(change, score, reasons[:3], correlation_tier(score))
        )
    return sorted(scored, key=lambda s: -s.score)


def completeness_note(changes: list[FileChange]) -> str | None:
    """Warn when diff data is missing for some changed files.

    Truncated patches and GitHub compare entries without a ``patch`` field
    mean we only know *that* a file changed, not *how* — so the ranking can
    miss culprits and must be flagged rather than trusted silently.
    """
    incomplete = [
        c
        for c in changes
        if c.patch_incomplete
        or (
            c.status in {"added", "modified"}
            and not c.added_lines
            and not c.removed_lines
        )
    ]
    if not incomplete:
        return None
    names = ", ".join(c.path for c in incomplete[:3])
    extra = f" and {len(incomplete) - 3} more" if len(incomplete) > 3 else ""
    return (
        f"{len(incomplete)} of {len(changes)} changed files have incomplete "
        f"diff data ({names}{extra}) — correlation may miss the true culprit."
    )


# ---------------------------------------------------------------------------
# Diff hunks -> indexable Documents (so changed code is retrievable context)
# ---------------------------------------------------------------------------

def changes_to_documents(
    changes: list[FileChange], max_lines: int = 60
) -> list[Document]:
    """Turn changed files into Documents for the vector store.

    Added lines carry their new-file line numbers in metadata, so evidence
    grounding and the UI can cite 'path:Lx-Ly' just like repo files.
    """
    docs: list[Document] = []
    for change in changes:
        if not (change.added_lines or change.removed_lines):
            continue
        parts = [f"# Change: {change.path} ({change.status}, {change.category})"]
        if change.old_path and change.old_path != change.path:
            parts.append(f"renamed from: {change.old_path}")
        if change.added_lines:
            parts.append("added:")
            parts.extend(f"{ln}: {t}" for ln, t in change.added_lines[:max_lines])
        if change.removed_lines:
            parts.append("removed:")
            parts.extend(change.removed_lines[: max_lines // 3])
        metadata = {
            "path": change.path,
            "source": "diff",
            "status": change.status,
            "category": change.category,
        }
        if change.added_lines:
            metadata["line_start"] = change.added_lines[0][0]
            metadata["line_end"] = change.added_lines[-1][0]
        docs.append(Document(page_content="\n".join(parts), metadata=metadata))
    return docs