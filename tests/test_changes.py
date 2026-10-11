"""Tests for change-aware analysis: parsing, classification, ranking."""
import pytest
from langchain_core.documents import Document

import src.graph.nodes as nodes_mod
import src.ingestion.github as github_mod
from src.changes import (
    FileChange,
    apply_patch_text,
    changes_to_documents,
    classify_change,
    completeness_note,
    correlate_changes,
    parse_unified_diff,
)
from src.evidence import extract_error_events
from src.graph.nodes import parse_log_node, retrieve_context_node
from src.ingestion.github import (
    load_github_compare,
    parse_github_compare_url,
)


SAMPLE_DIFF = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,4 @@
 import os
+import json
 print(json.dumps(x))
diff --git a/requirements.txt b/requirements.txt
index aaaaaaa..bbbbbbb 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,2 +1,3 @@
 flask==2.0
+django==5.0
diff --git a/old.py b/deleted.py
similarity index 90%
rename from old.py
rename to deleted.py
diff --git a/gone.sh b/gone.sh
deleted file mode 100644
index ccccccc..0000000
--- a/gone.sh
+++ /dev/null
@@ -1,2 +0,0 @@
-echo hi
-echo bye
"""


# --- Parsing -----------------------------------------------------------------

def test_parse_diff_statuses_and_paths():
    changes = {c.path: c for c in parse_unified_diff(SAMPLE_DIFF)}
    assert set(changes) == {
        "src/app.py", "requirements.txt", "deleted.py", "gone.sh",
    }
    assert changes["src/app.py"].status == "modified"
    assert changes["requirements.txt"].status == "modified"
    assert changes["deleted.py"].status == "renamed"
    assert changes["deleted.py"].old_path == "old.py"
    assert changes["gone.sh"].status == "deleted"


def test_parse_diff_line_numbers():
    changes = {c.path: c for c in parse_unified_diff(SAMPLE_DIFF)}
    assert changes["src/app.py"].added_lines == [(11, "import json")]
    assert changes["requirements.txt"].added_lines == [(2, "django==5.0")]
    assert changes["gone.sh"].removed_lines == ["echo hi", "echo bye"]


def test_parse_diff_git_show_header_skipped():
    text = (
        "commit abc123\nAuthor: Dev <dev@x>\n\n    fix things\n\n"
        + SAMPLE_DIFF
    )
    assert len(parse_unified_diff(text)) == 4


def test_parse_diff_headerless():
    text = (
        "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,2 @@\n a = 1\n+b = 2\n"
        "--- a/y.py\n+++ b/y.py\n@@ -1,1 +1,1 @@\n-c = 1\n+c = 2\n"
    )
    changes = parse_unified_diff(text)
    assert [c.path for c in changes] == ["x.py", "y.py"]
    assert changes[0].added_lines == [(2, "b = 2")]


def test_parse_diff_empty_text():
    assert parse_unified_diff("") == []
    assert parse_unified_diff("no diff here") == []


# --- Classification -----------------------------------------------------------

def test_classify_change():
    assert classify_change("requirements.txt") == "dependency"
    assert classify_change("package.json") == "dependency"
    assert classify_change("tests/test_app.py") == "test"
    assert classify_change("src/foo.test.js") == "test"
    assert classify_change(".github/workflows/ci.yml") == "ci"
    assert classify_change("Dockerfile") == "config"
    assert classify_change("config/settings.yaml") == "config"
    assert classify_change("README.md") == "docs"
    assert classify_change("src/app.py") == "source"


def test_parse_sets_category():
    changes = {c.path: c for c in parse_unified_diff(SAMPLE_DIFF)}
    assert changes["requirements.txt"].category == "dependency"
    assert changes["src/app.py"].category == "source"


# --- Ranking: obvious culprit -------------------------------------------------

def test_correlate_obvious_culprit_ranks_first():
    """The file cited by an error line outranks every other change."""
    changes = parse_unified_diff(SAMPLE_DIFF)
    log = (
        "eslint src\n"
        "src/app.py:11:7  error  'json' is not defined  no-undef\n"
        "Process completed with exit code 1.\n"
    )
    links = correlate_changes(changes, log, extract_error_events(log))
    assert links[0].change.path == "src/app.py"
    assert links[0].score >= 4
    assert links[0].tier == "strong"
    assert any("error line" in r for r in links[0].reasons)


def test_correlate_identifier_in_error_line():
    """A newly added identifier appearing in the traceback is a strong link."""
    change = FileChange(
        path="src/config.py",
        added_lines=[(12, "self.timeout = read_env('TIMEOUT')")],
    )
    log = "AttributeError: 'Config' object has no attribute 'timeout'\n"
    links = correlate_changes([change], log, extract_error_events(log))
    assert links[0].score >= 2
    assert any("timeout" in r for r in links[0].reasons)


# --- Ranking: several plausible changes ----------------------------------------

def test_correlate_ranks_among_several_plausible():
    """Event-cited files outrank category-only matches."""
    changes = parse_unified_diff(SAMPLE_DIFF)
    log = (
        "pytest tests/test_utils.py\n"
        "FAILED tests/test_utils.py::test_x - assert 1 == 2\n"
        "src/app.py:11: error undefined name\n"
        "Process completed with exit code 1.\n"
    )
    test_change = FileChange(path="tests/test_utils.py", category="test",
                             added_lines=[(1, "def test_x(): assert False")])
    links = correlate_changes(changes + [test_change], log,
                              extract_error_events(log))
    # Both event-cited files top the ranking; category bumps order them.
    assert {l.change.path for l in links[:2]} == {
        "tests/test_utils.py", "src/app.py",
    }
    assert links[0].tier == "strong"
    # requirements.txt has no link to this log -> sinks below the cited files.
    assert links[-1].score == 0


# --- Ranking: nothing related ---------------------------------------------------

def test_correlate_nothing_related_all_weak():
    changes = [
        FileChange(path="README.md", category="docs",
                   added_lines=[(1, "# New docs")]),
        FileChange(path="src/unrelated.py", category="source",
                   added_lines=[(1, "x = 1")]),
    ]
    log = "npm ERR! ERESOLVE unable to resolve dependency tree\n"
    links = correlate_changes(changes, log, extract_error_events(log))
    assert all(l.score == 0 for l in links)
    assert all(l.tier == "weak" for l in links)


def test_correlate_dependency_change_without_dep_error_stays_low():
    """Category match requires a matching log flavour — no free points."""
    changes = parse_unified_diff(SAMPLE_DIFF)
    log = "AssertionError: expected 200 got 500\n"
    links = correlate_changes(changes, log, extract_error_events(log))
    req = next(l for l in links if l.change.path == "requirements.txt")
    assert req.score == 0


def test_correlate_deleted_file_still_referenced():
    change = FileChange(path="tools/deploy.sh", status="deleted")
    log = "bash: tools/deploy.sh: No such file or directory\nexit code 127\n"
    links = correlate_changes([change], log, extract_error_events(log))
    assert links[0].score >= 4  # named in log + deleted-but-referenced


# --- Incomplete diffs -----------------------------------------------------------

def test_completeness_note_flags_missing_patch():
    changes = [
        FileChange(path="a.py", added_lines=[(1, "x")]),
        FileChange(path="big.py", patch_incomplete=True),
        FileChange(path="only_stat.py"),  # parsed header, no hunk data
    ]
    note = completeness_note(changes)
    assert "2 of 3" in note
    assert "big.py" in note


def test_completeness_note_silent_when_complete():
    changes = [FileChange(path="a.py", added_lines=[(1, "x")])]
    assert completeness_note(changes) is None
    assert completeness_note([]) is None


def test_completeness_note_ignores_clean_renames():
    renamed = FileChange(path="b.py", old_path="a.py", status="renamed")
    assert completeness_note([renamed]) is None


# --- Documents ------------------------------------------------------------------

def test_changes_to_documents():
    changes = parse_unified_diff(SAMPLE_DIFF)
    docs = {d.metadata["path"]: d for d in changes_to_documents(changes)}
    app_doc = docs["src/app.py"]
    assert app_doc.metadata["source"] == "diff"
    assert app_doc.metadata["line_start"] == 11
    assert "import json" in app_doc.page_content
    # deleted.py was renamed only — no hunk content, no document.
    assert "deleted.py" not in docs
    assert "gone.sh" in docs  # removed lines are still context


# --- GitHub compare -------------------------------------------------------------

def test_parse_github_compare_url():
    url = "https://github.com/o/r/compare/v1.2.3...feature/x"
    assert parse_github_compare_url(url) == ("o", "r", "v1.2.3", "feature/x")
    with pytest.raises(ValueError):
        parse_github_compare_url("https://github.com/o/r")
    with pytest.raises(ValueError):
        parse_github_compare_url("https://gitlab.com/o/r/compare/a...b")


class _FakeCompareResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_load_github_compare(monkeypatch):
    payload = {
        "files": [
            {
                "filename": "src/app.py",
                "status": "modified",
                "patch": "@@ -1,2 +1,3 @@\n a = 1\n+b = 2\n c = 3",
            },
            {"filename": "logo.png", "status": "modified"},  # binary: no patch
        ]
    }

    def fake_get(url, timeout, headers):
        assert "/compare/" in url
        assert "Authorization" not in headers  # no token configured
        return _FakeCompareResponse(200, payload)

    monkeypatch.setattr(github_mod.requests, "get", fake_get)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    changes = load_github_compare(
        "https://github.com/o/r/compare/main...feature"
    )
    by_path = {c.path: c for c in changes}
    assert by_path["src/app.py"].added_lines == [(2, "b = 2")]
    assert by_path["logo.png"].patch_incomplete is True


def test_load_github_compare_uses_token_from_env(monkeypatch):
    seen = {}

    def fake_get(url, timeout, headers):
        seen.update(headers)
        return _FakeCompareResponse(200, {"files": []})

    monkeypatch.setattr(github_mod.requests, "get", fake_get)
    monkeypatch.setenv("GITHUB_TOKEN", "tok-secret")
    load_github_compare("https://github.com/o/r/compare/a...b")
    assert seen["Authorization"] == "Bearer tok-secret"


def test_load_github_compare_error_never_leaks_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok-secret")
    monkeypatch.setattr(
        github_mod.requests, "get",
        lambda url, timeout, headers: _FakeCompareResponse(404),
    )
    with pytest.raises(RuntimeError) as excinfo:
        load_github_compare("https://github.com/o/r/compare/a...b")
    assert "tok-secret" not in str(excinfo.value)


def test_load_github_compare_404_hint_without_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        github_mod.requests, "get",
        lambda url, timeout, headers: _FakeCompareResponse(404),
    )
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        load_github_compare("https://github.com/o/r/compare/a...b")


# --- Graph integration -----------------------------------------------------------

def test_parse_log_node_returns_change_links():
    state = {
        "error_log": "src/app.py:11:7  error  'json' is not defined\n",
        "changes": parse_unified_diff(SAMPLE_DIFF),
    }
    result = parse_log_node(state)
    links = result["change_links"]
    assert links[0].change.path == "src/app.py"
    assert links[0].score >= 4


def test_retrieve_context_queries_correlated_changes(monkeypatch):
    captured: list = []

    def fake_retrieve(store, queries, top_k=None):
        captured.extend(queries)
        return []

    monkeypatch.setattr(nodes_mod, "retrieve", fake_retrieve)
    state = {
        "error_log": "src/app.py:11:7  error  'json' is not defined\n",
        "changes": parse_unified_diff(SAMPLE_DIFF),
        "vectorstore": object(),
    }
    state.update(parse_log_node(state))
    retrieve_context_node(state)
    assert any("src/app.py" in q for q in captured)
    # Zero-score changes are not used as queries.
    assert not any("gone.sh" in q for q in captured)


def test_analyze_prompt_lists_recent_changes(
    monkeypatch, fake_llm, sample_log
):
    captured: list = []

    class _Cap:
        def with_structured_output(self, schema):
            from langchain_core.runnables import RunnableLambda
            return RunnableLambda(
                lambda p: (captured.append(p.to_string()), fake_llm.analysis)[1]
            )

    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: _Cap())
    nodes_mod.analyze_failure_node(
        {
            "error_log": sample_log,
            "retrieved_docs": [],
            "changes": parse_unified_diff(SAMPLE_DIFF),
            "change_links": [],
        }
    )
    assert "## Recent Changes" in captured[0]
    assert "src/app.py" in captured[0]
    assert "requirements.txt" in captured[0]
