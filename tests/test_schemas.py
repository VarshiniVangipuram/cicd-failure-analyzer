"""Tests for the FailureAnalysis schema."""
import pytest
from pydantic import ValidationError

from src.schemas import (
    AlternativeCause,
    EvidenceItem,
    FailureAnalysis,
    SuspectChange,
)


def test_valid_analysis():
    a = FailureAnalysis(
        failure_type="dependency_error",
        root_cause="npm cannot resolve dependency tree",
        evidence=["npm ERR! ERESOLVE"],
        relevant_files=["package.json"],
        recommended_fixes=["Pin versions", "Run npm install --legacy-peer-deps"],
        confidence=0.8,
        confidence_level="high",
    )
    assert a.confidence == 0.8


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        FailureAnalysis(
            failure_type="x",
            root_cause="x",
            confidence=1.5,
            confidence_level="high",
        )


def test_confidence_level_literal():
    with pytest.raises(ValidationError):
        FailureAnalysis(
            failure_type="x",
            root_cause="x",
            confidence=0.5,
            confidence_level="certain",
        )


def test_new_fields_have_safe_defaults():
    a = FailureAnalysis(
        failure_type="x",
        root_cause="x",
        confidence=0.5,
        confidence_level="medium",
    )
    assert a.alternative_causes == []
    assert a.evidence_sufficient is True
    assert a.confidence_rationale == ""


def test_alternative_cause_and_evidence_item():
    alt = AlternativeCause(summary="flaky test", supporting_evidence=["L9 FAILED"])
    assert alt.summary == "flaky test"
    item = EvidenceItem(kind="file", reference="src/app.py:3-5", excerpt="x = 1")
    assert item.verified is True


def test_evidence_item_kind_literal():
    with pytest.raises(ValidationError):
        EvidenceItem(kind="internet", excerpt="x")


def test_suspect_changes_default_and_shape():
    a = FailureAnalysis(
        failure_type="x",
        root_cause="x",
        confidence=0.5,
        confidence_level="medium",
    )
    assert a.suspect_changes == []
    a.suspect_changes.append(
        SuspectChange(change="requirements.txt — added django==5.0",
                      why="dep manifest changed and the log shows ERESOLVE")
    )
    assert a.suspect_changes[0].why.startswith("dep manifest")
