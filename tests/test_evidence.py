"""Tests for src.evidence — extraction, ranking, grounding, confidence."""
from langchain_core.documents import Document

from src.evidence import (
    ErrorEvent,
    extract_error_events,
    ground_evidence,
    match_error,
    rank_error_events,
    reconcile_confidence,
)
from src.schemas import EvidenceItem, FailureAnalysis


def _analysis(**overrides) -> FailureAnalysis:
    base = dict(
        failure_type="build_error",
        root_cause="something broke",
        confidence=0.9,
        confidence_level="high",
    )
    base.update(overrides)
    return FailureAnalysis(**base)


LOG = (
    "[10:00:01] Installing dependencies...\n"
    "[10:00:02] npm ci\n"
    "[10:00:10] npm ERR! ERESOLVE unable to resolve dependency tree\n"
    "[10:00:12] 12:5  error  'foo' is not defined  no-undef\n"
    "[10:00:13] Process completed with exit code 1.\n"
)


# --- match_error -------------------------------------------------------------

def test_match_error_classification():
    assert match_error("AssertionError: expected 200")[0] == "exception"
    assert match_error("Traceback (most recent call last):")[0] == "traceback"
    assert match_error("Process completed with exit code 1.")[0] == "exit_code"
    assert match_error("everything is fine") is None


# --- extract_error_events ----------------------------------------------------

def test_extract_events_line_numbers_and_kinds():
    events = extract_error_events(LOG)
    assert [e.line_number for e in events] == [3, 4, 5]
    assert events[0].kind == "error"
    assert events[-1].kind == "exit_code"


def test_extract_events_empty_log():
    assert extract_error_events("") == []


def test_extract_events_handles_multiple_errors():
    log = (
        "step: flake8 .\n"
        "./app.py:12:5: error 'foo' undefined\n"   # L2 error
        "step: pytest\n"
        "FAILED tests/test_app.py::test_x - assert 1 == 2\n"  # L5 failed
        "Process completed with exit code 1.\n"               # L6 exit_code
    )
    events = extract_error_events(log)
    kinds = {e.kind for e in events}
    assert len(events) == 3
    assert "error" in kinds and "failed" in kinds and "exit_code" in kinds


# --- rank_error_events -------------------------------------------------------

def test_rank_prefers_highest_severity_not_last_line():
    """'exit code 1' is a symptom — a real error line must outrank it."""
    events = extract_error_events(LOG)
    ranked = rank_error_events(events)
    assert ranked[0].line_number == 3
    assert ranked[-1].kind == "exit_code"


def test_rank_earlier_line_wins_ties():
    log = "npm ERR! first\nall ok\nnpm ERR! second\n"
    ranked = rank_error_events(extract_error_events(log))
    assert ranked[0].text == "npm ERR! first"


def test_rank_severity_beats_position():
    log = (
        "error: artifact download hiccup\n"          # L1 error (0.7)
        "AssertionError: expected 200 got 500\n"     # L2 exception (0.95)
        "Process completed with exit code 1.\n"      # L3 exit_code (0.5)
    )
    ranked = rank_error_events(extract_error_events(log))
    assert ranked[0].line_number == 2


def test_rank_is_stable_and_does_not_mutate_input():
    events = extract_error_events(LOG)
    snapshot = list(events)
    rank_error_events(events)
    assert events == snapshot


# --- ground_evidence ----------------------------------------------------------

def test_ground_log_evidence_gets_line_number():
    a = _analysis(evidence=["npm ERR! ERESOLVE unable to resolve dependency tree"])
    items = ground_evidence(a, LOG, [])
    assert len(items) == 1
    assert items[0].kind == "log"
    assert items[0].reference == "log line 3"
    assert items[0].verified


def test_ground_log_evidence_fuzzy_match():
    """Paraphrased claims still resolve to a line via token overlap."""
    a = _analysis(evidence=["ERESOLVE resolve dependency tree npm"])
    items = ground_evidence(a, LOG, [])
    assert items[0].verified
    assert items[0].reference == "log line 3"


def test_ground_file_evidence_gets_path_and_line():
    doc = Document(
        page_content="# File: package.json\n{\n  \"dependencies\": {\n"
                     '    "react": "^18.0.0"\n  }\n}',
        metadata={"path": "package.json", "line_start": 1, "line_end": 5},
    )
    a = _analysis(evidence=['"react": "^18.0.0"'])
    items = ground_evidence(a, "no matching log content", [doc])
    assert items[0].kind == "file"
    assert items[0].reference == "package.json:3"
    assert items[0].verified


def test_ground_file_evidence_chunk_range():
    """A paraphrased claim overlapping a whole chunk cites its line range."""
    doc = Document(
        page_content="# File: ci.yml\nname: ci\nsteps:\n  - run: npm ci",
        metadata={"path": "ci.yml", "line_start": 10, "line_end": 12},
    )
    a = _analysis(evidence=["name: ci steps:"])
    items = ground_evidence(a, "unrelated log", [doc])
    assert items[0].verified
    assert items[0].reference == "ci.yml:10-12"


def test_ground_unverifiable_claim_flagged():
    a = _analysis(evidence=["the deployment key is probably expired"])
    items = ground_evidence(a, LOG, [])
    assert len(items) == 1
    assert not items[0].verified
    assert items[0].reference == ""


def test_ground_empty_evidence():
    assert ground_evidence(_analysis(evidence=[]), LOG, []) == []


# --- Module-shadowing regression --------------------------------------------
# Log + repo matching the reported bug: traceback cites root-level json.py,
# which shadows stdlib json. All claims verified but every item was
# classified 'log', so reconcile_confidence wrongly reported missing file
# corroboration.

SHADOW_LOG = (
    "python -m pytest\n"
    "Traceback (most recent call last):\n"
    '  File "runner.py", line 5, in <module>\n'
    "    import json\n"
    '  File "C:\\project\\json.py", line 1, in <module>\n'
    "    import json\n"
    "AttributeError: partially initialized module 'json' has no attribute 'dumps'\n"
)

SHADOW_DOC = Document(
    page_content="# File: json.py\nimport json\nprint(json.dumps({'a': 1}))",
    metadata={"path": "json.py", "line_start": 1, "line_end": 2},
)


def test_ground_traceback_frame_citing_repo_file_is_file_evidence():
    """A traceback line pointing at a retrieved file counts as file evidence."""
    a = _analysis(evidence=['File "C:\\project\\json.py", line 1, in <module>'])
    items = ground_evidence(a, SHADOW_LOG, [SHADOW_DOC])
    assert len(items) == 1
    assert items[0].verified
    assert items[0].kind == "file"
    assert "log line 5" in items[0].reference
    assert "json.py:1" in items[0].reference


def test_ground_traceback_to_missing_file_stays_log_evidence():
    """Traceback frames citing files we didn't retrieve remain log evidence."""
    a = _analysis(evidence=['File "runner.py", line 5, in <module>'])
    items = ground_evidence(a, SHADOW_LOG, [SHADOW_DOC])  # only json.py present
    assert items[0].verified
    assert items[0].kind == "log"
    assert items[0].reference == "log line 3"


def test_shadowing_scenario_no_false_corroboration_warning():
    """Full regression: verified claims + retrieved json.py => no warning."""
    a = _analysis(
        evidence=[
            "Traceback (most recent call last):",
            'File "C:\\project\\json.py", line 1, in <module>',
            "AttributeError: partially initialized module 'json' "
            "has no attribute 'dumps'",
            "json.py shadows the standard library json module",
        ]
    )
    items = ground_evidence(a, SHADOW_LOG, [SHADOW_DOC])
    assert len(items) == 4
    assert all(i.verified for i in items)
    assert any(i.kind == "file" for i in items)

    events = extract_error_events(SHADOW_LOG)
    score, level, notes = reconcile_confidence(a, items, [SHADOW_DOC], events)
    assert not any("corroborates" in n for n in notes)
    assert level == "high"


def test_shadowing_scenario_without_repo_file_still_warns():
    """Same diagnosis without json.py in context -> warning must still fire."""
    a = _analysis(
        evidence=['File "C:\\project\\json.py", line 1, in <module>']
    )
    unrelated = Document(
        page_content="# File: other.py\nx = 1",
        metadata={"path": "other.py", "line_start": 1, "line_end": 1},
    )
    items = ground_evidence(a, SHADOW_LOG, [unrelated])
    assert items[0].kind == "log"

    events = extract_error_events(SHADOW_LOG)
    _, _, notes = reconcile_confidence(a, items, [unrelated], events)
    assert any("corroborates" in n for n in notes)


# --- reconcile_confidence ------------------------------------------------------

def test_confidence_collapses_without_any_evidence():
    a = _analysis(confidence=0.95, evidence=[])
    score, level, notes = reconcile_confidence(a, [], [], [])
    assert score <= 0.3
    assert level == "low"
    assert notes  # reasons are surfaced for the UI


def test_confidence_capped_without_project_context():
    a = _analysis(
        evidence=["npm ERR! ERESOLVE unable to resolve dependency tree"]
    )
    items = ground_evidence(a, LOG, [])
    events = extract_error_events(LOG)
    score, level, notes = reconcile_confidence(a, items, [], events)
    assert score == 0.6            # log evidence verified, no project context
    assert level == "medium"
    assert any("project context" in n for n in notes)


def test_confidence_capped_when_model_flags_insufficient():
    a = _analysis(evidence_sufficient=False)
    items = [
        EvidenceItem(kind="log", reference="log line 3", excerpt="e"),
        EvidenceItem(kind="file", reference="a.py:1", excerpt="x"),
        EvidenceItem(kind="file", reference="b.py:2", excerpt="y"),
    ]
    score, _, notes = reconcile_confidence(a, items, [object()], [ErrorEvent(1, "t", "error", 0.7)])
    assert score <= 0.5
    assert any("insufficient" in n for n in notes)


def test_confidence_unchanged_with_strong_evidence():
    a = _analysis(confidence=0.9)
    items = [
        EvidenceItem(kind="log", reference="log line 1", excerpt="e1"),
        EvidenceItem(kind="file", reference="a.py:1-3", excerpt="x"),
    ]
    docs = [Document(page_content="x", metadata={"path": "a.py"})]
    events = [ErrorEvent(1, "boom error", "error", 0.7)]
    score, level, notes = reconcile_confidence(a, items, docs, events)
    assert score == 0.9
    assert level == "high"
    assert notes == []


def test_confidence_level_always_matches_score():
    a = _analysis(confidence=0.85, confidence_level="high", evidence=[])
    score, level, _ = reconcile_confidence(a, [], [], [])
    assert level == "low"   # recomputed from the capped score, not the label
