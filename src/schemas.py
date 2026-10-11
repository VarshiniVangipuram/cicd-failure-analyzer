"""Pydantic models describing the analyzer's structured output."""
from typing import Literal

from pydantic import BaseModel, Field


class AlternativeCause(BaseModel):
    """A secondary hypothesis, kept separate from the primary root cause."""

    summary: str = Field(
        description="Short description of another plausible cause."
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Log lines or file references supporting this cause.",
    )


class SuspectChange(BaseModel):
    """A recent change assessed as *likely related* to the failure.

    Correlation, not proven causation — ``why`` must cite the link (log
    reference, matching error kind) rather than assert guilt.
    """

    change: str = Field(
        description="The suspect change, e.g. 'requirements.txt — added django==5.0'."
    )
    why: str = Field(
        description="Why this change is plausibly related, tied to log evidence."
    )


class EvidenceItem(BaseModel):
    """A piece of supporting evidence grounded to a concrete location.

    Produced deterministically by ``src.evidence.ground_evidence`` — not by
    the LLM — so references are only marked verified when they actually
    appear in the log or retrieved files.
    """

    kind: Literal["log", "file"] = Field(
        description="Whether the evidence comes from the log or a project file."
    )
    reference: str = Field(
        default="",
        description=(
            "Precise location, e.g. 'log line 12' or 'src/app.py:34-40'. "
            "Empty when the claim could not be verified."
        ),
    )
    excerpt: str = Field(description="The cited log line or code snippet.")
    verified: bool = Field(
        default=True,
        description="False when the model's claim matched no real artifact.",
    )


class FailureAnalysis(BaseModel):
    """Structured diagnosis of a CI/CD pipeline failure."""

    failure_type: str = Field(
        description=(
            "Short category of the failure, e.g. 'dependency_error', "
            "'test_failure', 'build_error', 'lint_error', 'deploy_error', "
            "'config_error', 'timeout', 'unknown'"
        )
    )
    root_cause: str = Field(
        description="A concise explanation of the most likely root cause."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Log lines or code snippets that support the diagnosis.",
    )
    relevant_files: list[str] = Field(
        default_factory=list,
        description="File paths in the project most related to the failure.",
    )
    alternative_causes: list[AlternativeCause] = Field(
        default_factory=list,
        description=(
            "Other plausible causes when the log contains multiple errors, "
            "ordered by likelihood below the primary root cause."
        ),
    )
    suspect_changes: list[SuspectChange] = Field(
        default_factory=list,
        description=(
            "Recent changes likely related to the failure — correlation, "
            "not proven causation. Empty when no diff was provided or no "
            "change plausibly relates."
        ),
    )
    recommended_fixes: list[str] = Field(
        default_factory=list,
        description="Ordered, actionable steps to fix the failure.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the diagnosis between 0.0 and 1.0.",
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        description="Human-readable confidence band derived from the score."
    )
    confidence_rationale: str = Field(
        default="",
        description=(
            "One-sentence justification for the confidence score, noting "
            "which evidence is present or missing."
        ),
    )
    evidence_sufficient: bool = Field(
        default=True,
        description=(
            "False when the log and project context do not contain enough "
            "information for a reliable diagnosis."
        ),
    )
