"""End-to-end test: full LangGraph pipeline with a real (fake-embedding)
vector store and a mocked Gemini LLM — no network or API calls."""
import pytest

import src.graph.nodes as nodes_mod
from src.changes import FileChange
from src.graph import build_graph
from src.rag.indexer import build_vectorstore
from src.schemas import FailureAnalysis, SuspectChange


def test_full_pipeline_with_context(
    monkeypatch, fake_embeddings, fake_llm, sample_docs, sample_log
):
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)

    graph = build_graph()
    result = graph.invoke(
        {"error_log": sample_log, "vectorstore": store}
    )

    # Every stage populated the shared state.
    assert result["error_signatures"]
    assert result["retrieved_docs"]
    assert result["error_events"]
    assert result["primary_error"] is not None
    assert result["primary_error"].line_number > 0
    analysis = result["analysis"]
    assert isinstance(analysis, FailureAnalysis)
    assert analysis.failure_type == "dependency_error"
    assert analysis.recommended_fixes
    assert 0.0 <= analysis.confidence <= 1.0
    # Evidence was grounded to real log lines.
    assert result["evidence_items"]
    assert any(
        e.verified and e.reference.startswith("log line")
        for e in result["evidence_items"]
    )


def test_full_pipeline_without_context(
    monkeypatch, fake_llm, sample_log
):
    """No project context — graph still produces a diagnosis."""
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)

    graph = build_graph()
    result = graph.invoke({"error_log": sample_log, "vectorstore": None})

    assert result["retrieved_docs"] == []
    assert result["analysis"].failure_type == "dependency_error"
    # No project context -> confidence must be capped and explained.
    assert result["analysis"].confidence <= 0.6
    assert result["confidence_notes"]


def test_full_pipeline_with_user_question(
    monkeypatch, fake_embeddings, fake_llm, sample_docs, sample_log
):
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)

    graph = build_graph()
    result = graph.invoke(
        {
            "error_log": sample_log,
            "user_question": "What is the safest fix to apply first?",
            "vectorstore": store,
        }
    )

    analysis = result["analysis"]
    assert isinstance(analysis, FailureAnalysis)
    assert analysis.recommended_fixes


def test_full_pipeline_with_changes(
    monkeypatch, fake_embeddings, fake_llm, sample_docs, sample_log
):
    """A provided diff flows through correlation into the prompt."""
    fake_llm.analysis = FailureAnalysis(
        failure_type="dependency_error",
        root_cause="requirements.txt added django==5.0 which broke the build",
        evidence=["npm ERR! ERESOLVE unable to resolve dependency tree"],
        relevant_files=["requirements.txt"],
        recommended_fixes=["Revert the new dependency"],
        confidence=0.9,
        confidence_level="high",
        suspect_changes=[
            SuspectChange(
                change="requirements.txt — added django==5.0",
                why="the dependency manifest changed and the log shows ERESOLVE",
            )
        ],
    )
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)
    changes = [
        FileChange(
            path="requirements.txt",
            status="modified",
            category="dependency",
            added_lines=[(2, "django==5.0")],
        )
    ]

    graph = build_graph()
    result = graph.invoke(
        {
            "error_log": sample_log,
            "vectorstore": store,
            "changes": changes,
        }
    )

    assert result["change_links"]
    assert result["change_links"][0].change.path == "requirements.txt"
    assert result["change_links"][0].score >= 2  # dep manifest + ERESOLVE log
    assert result["analysis"].suspect_changes[0].change.startswith(
        "requirements.txt"
    )


def test_full_pipeline_api_failure(
    monkeypatch, fake_embeddings, sample_docs, sample_log, fake_llm
):
    fake_llm.exc = ConnectionError("rate limited")
    monkeypatch.setattr(nodes_mod, "get_llm", lambda *a, **k: fake_llm)
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)

    graph = build_graph()
    with pytest.raises(ConnectionError, match="rate limited"):
        graph.invoke({"error_log": sample_log, "vectorstore": store})
