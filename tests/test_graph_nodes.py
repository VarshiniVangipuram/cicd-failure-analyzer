"""Tests for graph nodes — the Gemini call is replaced by FakeLLM."""
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import src.graph.nodes as nodes_mod
from src.graph.nodes import (
    _format_context,
    analyze_failure_node,
    extract_error_signatures,
    parse_log_node,
    retrieve_context_node,
)
from src.llm import get_llm
from src.rag.indexer import build_vectorstore


class _CaptureLLM:
    """Fake LLM that records the formatted prompt it receives."""

    def __init__(self, analysis, captured: list):
        self.analysis = analysis
        self.captured = captured

    def with_structured_output(self, schema):
        from langchain_core.runnables import RunnableLambda

        def _run(prompt_value):
            self.captured.append(prompt_value.to_string())
            return self.analysis

        return RunnableLambda(_run)


SAMPLE_LOG = """
[10:00:01] Installing dependencies...
[10:00:02] npm ci
[10:00:10] npm ERR! code ERESOLVE
[10:00:10] npm ERR! ERESOLVE unable to resolve dependency tree
[10:00:11] Process completed with exit code 1.
"""


def test_extract_error_signatures_finds_errors():
    sigs = extract_error_signatures(SAMPLE_LOG)
    assert any("ERESOLVE" in s for s in sigs)
    assert any("exit code 1" in s for s in sigs)


def test_extract_error_signatures_fallback_last_lines():
    sigs = extract_error_signatures("line one\nline two\nline three")
    assert sigs == ["line one", "line two", "line three"]


def test_parse_log_node():
    result = parse_log_node({"error_log": SAMPLE_LOG})
    assert "error_signatures" in result
    assert len(result["error_signatures"]) > 0
    # Structured events are ranked; the first ERR! line is the primary suspect.
    assert result["primary_error"] == result["error_events"][0]
    assert result["primary_error"].line_number == 4


def test_parse_log_node_multi_error_prioritizes_severity():
    log = (
        "warn: cache miss\n"
        "error: artifact download hiccup\n"          # L2 — generic error
        "AssertionError: expected 200 got 500\n"     # L3 — exception
        "Process completed with exit code 1.\n"      # L4 — symptom
    )
    result = parse_log_node({"error_log": log})
    assert len(result["error_events"]) == 3
    assert result["primary_error"].line_number == 3
    assert result["primary_error"].kind == "exception"


def test_extract_error_signatures_empty_log():
    assert extract_error_signatures("") == []


def test_retrieve_context_node_without_store():
    result = retrieve_context_node({"error_log": "x", "error_signatures": ["a"]})
    assert result == {"retrieved_docs": []}


def test_retrieve_context_node_with_store(
    fake_embeddings, sample_docs, sample_log
):
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)
    state = {
        "error_log": sample_log,
        "error_signatures": extract_error_signatures(sample_log),
        "vectorstore": store,
    }
    result = retrieve_context_node(state)
    assert len(result["retrieved_docs"]) > 0


def test_retrieve_context_adds_question_query(monkeypatch):
    captured_queries: list = []

    def fake_retrieve(store, queries, top_k=None):
        captured_queries.extend(queries)
        return []

    monkeypatch.setattr(nodes_mod, "retrieve", fake_retrieve)
    retrieve_context_node(
        {
            "error_signatures": ["npm ERR!"],
            "user_question": "Is this a dependency conflict?",
            "vectorstore": object(),  # non-None is enough; retrieve is stubbed
        }
    )
    assert "Is this a dependency conflict?" in captured_queries


def test_retrieve_context_ignores_blank_question(monkeypatch):
    captured_queries: list = []

    def fake_retrieve(store, queries, top_k=None):
        captured_queries.extend(queries)
        return []

    monkeypatch.setattr(nodes_mod, "retrieve", fake_retrieve)
    retrieve_context_node(
        {
            "error_signatures": ["npm ERR!"],
            "user_question": "   ",
            "vectorstore": object(),
        }
    )
    # only the signature + generic query — no blank question appended
    assert captured_queries == [
        "npm ERR!",
        "CI CD pipeline configuration build test deploy error",
    ]


def test_format_context_empty():
    assert _format_context([]) == "(no project context provided)"


def test_format_context_includes_paths():
    docs = [Document(page_content="body", metadata={"path": "a.py"})]
    out = _format_context(docs)
    assert "--- a.py ---" in out


def test_format_context_truncates():
    docs = [
        Document(page_content="x" * 1500, metadata={"path": f"f{i}.py"})
        for i in range(20)
    ]
    out = _format_context(docs, max_chars=10_000)
    assert len(out) <= 10_000 + 100


# --- analyze_failure_node (mocked Gemini) ------------------------------------

def test_analyze_failure_node_success(
    monkeypatch, fake_llm, sample_log
):
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    result = analyze_failure_node(
        {"error_log": sample_log, "error_signatures": ["npm ERR!"], "retrieved_docs": []}
    )
    analysis = result["analysis"]
    assert analysis.failure_type == "dependency_error"
    assert 0.0 <= analysis.confidence <= 1.0


def test_analyze_failure_node_api_failure(monkeypatch, sample_log, fake_llm):
    fake_llm.exc = RuntimeError("Gemini API unavailable")
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    with pytest.raises(RuntimeError, match="Gemini API unavailable"):
        analyze_failure_node({"error_log": sample_log})


def test_analyze_failure_node_grounds_evidence(
    monkeypatch, fake_llm, sample_log
):
    """Model-cited evidence is verified against the log and gets a line ref."""
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    state = {"error_log": sample_log, "retrieved_docs": []}
    state.update(parse_log_node(state))
    result = analyze_failure_node(state)

    items = result["evidence_items"]
    assert items and items[0].verified
    assert items[0].kind == "log"
    assert items[0].reference == "log line 4"   # the ERESOLVE line


def test_analyze_failure_node_caps_confidence_without_context(
    monkeypatch, fake_llm, sample_log
):
    """With no project files, the model's 0.85 self-score is capped down."""
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    state = {"error_log": sample_log, "retrieved_docs": []}
    state.update(parse_log_node(state))
    result = analyze_failure_node(state)
    assert result["analysis"].confidence == 0.6
    assert result["analysis"].confidence_level == "medium"
    assert result["confidence_notes"]


def test_analyze_failure_prompt_lists_line_numbered_events(
    monkeypatch, fake_llm, sample_log
):
    captured: list = []
    monkeypatch.setattr(
        nodes_mod, "get_llm", lambda *a, **k: _CaptureLLM(fake_llm.analysis, captured)
    )
    state = {"error_log": sample_log, "retrieved_docs": []}
    state.update(parse_log_node(state))
    analyze_failure_node(state)
    assert "L4 [error]" in captured[0]
    assert "primary suspect" in captured[0]


def test_analyze_failure_node_includes_user_question(
    monkeypatch, fake_llm, sample_log
):
    captured: list = []
    monkeypatch.setattr(
        nodes_mod, "get_llm", lambda *a, **k: _CaptureLLM(fake_llm.analysis, captured)
    )
    analyze_failure_node(
        {
            "error_log": sample_log,
            "user_question": "Is this a dependency conflict?",
            "retrieved_docs": [],
        }
    )
    assert len(captured) == 1
    assert "Is this a dependency conflict?" in captured[0]
    assert "User's Investigation Request" in captured[0]


def test_analyze_failure_node_no_question_matches_default(
    monkeypatch, fake_llm, sample_log
):
    captured: list = []
    monkeypatch.setattr(
        nodes_mod, "get_llm", lambda *a, **k: _CaptureLLM(fake_llm.analysis, captured)
    )
    analyze_failure_node(
        {"error_log": sample_log, "user_question": "", "retrieved_docs": []}
    )
    analyze_failure_node(
        {"error_log": sample_log, "retrieved_docs": []}  # key absent entirely
    )
    assert len(captured) == 2
    assert captured[0] == captured[1]  # empty == absent
    assert "Investigation Request" not in captured[0]


def test_analyze_failure_node_missing_api_key(monkeypatch, sample_log):
    monkeypatch.setattr(
        "src.llm.settings",
        SimpleNamespace(google_api_key="", gemini_model="m"),
    )
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_llm()


def test_get_llm_api_key_override(monkeypatch):
    monkeypatch.setattr(
        "src.llm.settings",
        SimpleNamespace(google_api_key="", gemini_model="gemini-3.8-flash"),
    )
    llm = get_llm(api_key="test-key")
    assert "gemini-3.8-flash" in llm.model
