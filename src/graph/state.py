"""State shared across LangGraph nodes."""
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.changes import FileChange, ScoredChange
from src.evidence import ErrorEvent
from src.schemas import EvidenceItem, FailureAnalysis


class AnalysisState(TypedDict, total=False):
    error_log: str                       # raw CI/CD log text from the user
    user_question: str                   # optional investigation focus
    changes: list[FileChange]            # recent code changes (set before invoke)
    error_signatures: list[str]          # extracted error lines (parse node)
    error_events: list[ErrorEvent]       # ranked, line-numbered error events
    primary_error: ErrorEvent | None     # top-ranked root-cause candidate
    change_links: list[ScoredChange]     # changes scored against the log
    vectorstore: Chroma                  # ephemeral index (set before invoke)
    retrieved_docs: list[Document]       # evidence docs (retrieve node)
    analysis: FailureAnalysis | None     # final structured diagnosis
    evidence_items: list[EvidenceItem]   # verified, location-tagged evidence
    confidence_notes: list[str]          # why confidence was capped, if it was
