"""CI/CD Failure Analyzer — Streamlit entry point.

Run with:  streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

from src.changes import (
    changes_to_documents,
    completeness_note,
    parse_unified_diff,
)
from src.graph import build_graph
from src.ingestion import (
    load_github_compare,
    load_github_repo,
    load_uploaded_files,
    load_zip_bytes,
)
from src.rag.indexer import build_vectorstore
from src.schemas import FailureAnalysis

load_dotenv()

st.set_page_config(
    page_title="CI/CD Failure Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIDENCE_ICON = {"low": "🔴", "medium": "🟡", "high": "🟢"}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2.5rem; max-width: 1100px; }

        .section-label {
            font-size: .78rem; font-weight: 700; letter-spacing: .1em;
            text-transform: uppercase; opacity: .55; margin-bottom: -.4rem;
        }
        .hero-sub {
            font-size: 1.05rem; opacity: .7; margin-top: -.8rem;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 10px; padding: .9rem 1.1rem;
            background: rgba(128,128,128,.05);
        }
        .fix-card {
            border: 1px solid rgba(128,128,128,.25);
            border-left: 4px solid #4A90D9;
            border-radius: 8px; padding: .7rem 1rem; margin-bottom: .55rem;
            background: rgba(128,128,128,.04);
        }
        .fix-num { font-weight: 700; color: #4A90D9; margin-right: .4rem; }
        .file-pill {
            display: inline-block; border: 1px solid rgba(128,128,128,.3);
            border-radius: 999px; padding: .15rem .8rem; margin: .15rem .25rem .15rem 0;
            font-family: monospace; font-size: .85rem;
        }
        ::placeholder {
            opacity: .4 !important;
            filter: blur(.3px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")

        if os.getenv("GOOGLE_API_KEY"):
            # Key comes from .env — never render it into the page.
            st.success("Gemini API key loaded from environment", icon="✅")
        else:
            st.warning("No API key configured", icon="⚠️")
            api_key = st.text_input(
                "Google Gemini API Key",
                type="password",
                help="Get a key at https://aistudio.google.com/app/apikey",
            )
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                st.rerun()

        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Provide project context (repo, ZIP, or files)\n"
            "2. Paste the failed pipeline log\n"
            "3. Optionally focus the investigation\n"
            "4. Get a diagnosis with evidence and fixes"
        )
        st.divider()
        st.caption("LangGraph · ChromaDB · sentence-transformers · Gemini")


# ---------------------------------------------------------------------------
# Header + inputs
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.title("🔍 CI/CD Failure Analyzer")
    st.markdown(
        '<p class="hero-sub">AI-assisted root-cause analysis for failed '
        "pipelines — powered by RAG over your codebase.</p>",
        unsafe_allow_html=True,
    )


def section(number: int, title: str, caption: str | None = None) -> None:
    st.markdown(
        f'<div class="section-label">Step {number}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------
def gather_documents(github_url: str, zip_file, individual_files):
    """Collect Documents from every provided context source."""
    docs = []
    warnings = []

    if github_url.strip():
        with st.spinner("Downloading GitHub repository…"):
            try:
                docs.extend(load_github_repo(github_url.strip()))
            except Exception as exc:
                warnings.append(f"GitHub load failed: {exc}")

    if zip_file is not None:
        with st.spinner("Extracting ZIP archive…"):
            try:
                docs.extend(load_zip_bytes(zip_file.getvalue()))
            except Exception as exc:
                warnings.append(f"ZIP load failed: {exc}")

    if individual_files:
        pairs = [(f.name, f.getvalue()) for f in individual_files]
        docs.extend(load_uploaded_files(pairs))

    return docs, warnings


def gather_changes(diff_text: str, diff_file, compare_url: str):
    """Parse change context from pasted/uploaded diff or a compare URL.

    Returns (changes, diff_documents, warnings).
    """
    changes = []
    warnings = []

    text = diff_text.strip()
    if not text and diff_file is not None:
        text = diff_file.getvalue().decode("utf-8", errors="replace").strip()
    if text:
        changes.extend(parse_unified_diff(text))

    if compare_url.strip():
        try:
            changes.extend(load_github_compare(compare_url.strip()))
        except Exception as exc:
            warnings.append(f"Compare fetch failed: {exc}")

    # Same file via two sources — keep the first occurrence.
    seen, unique = set(), []
    for change in changes:
        if change.path not in seen:
            seen.add(change.path)
            unique.append(change)
    return unique, changes_to_documents(unique), warnings


# ---------------------------------------------------------------------------
# Analysis run
# ---------------------------------------------------------------------------
def run_analysis(error_log: str, question: str, docs, changes=None) -> dict | None:
    """Index context, invoke the LangGraph pipeline, return final state."""
    vectorstore = None
    if docs:
        with st.spinner("Embedding project context…"):
            vectorstore = build_vectorstore(docs)

    graph = build_graph()
    with st.spinner("Analyzing failure…"):
        return graph.invoke(
            {
                "error_log": error_log,
                "user_question": question,
                "vectorstore": vectorstore,
                "changes": changes or [],
            }
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_results(result: dict, doc_count: int) -> None:
    analysis: FailureAnalysis = result["analysis"]
    retrieved = result.get("retrieved_docs", [])
    events = result.get("error_events", [])
    evidence_items = result.get("evidence_items", [])
    confidence_notes = result.get("confidence_notes", [])
    primary = result.get("primary_error")
    changes = result.get("changes", [])
    change_links = result.get("change_links", [])

    st.divider()
    st.markdown(
        '<div class="section-label">Result</div>', unsafe_allow_html=True
    )
    st.markdown("### Diagnosis")
    verified_count = sum(1 for e in evidence_items if e.verified)
    changes_part = f"{len(changes)} changed files analyzed · " if changes else ""
    st.caption(
        f"{doc_count} files indexed · "
        f"{changes_part}"
        f"{len(events)} error events detected · "
        f"{verified_count}/{len(evidence_items)} evidence items verified · "
        f"{len(retrieved)} context chunks retrieved"
    )

    if changes:
        note = completeness_note(changes)
        if note:
            st.info(note, icon="ℹ️")

    if confidence_notes:
        st.warning(
            "Confidence is limited — " + "; ".join(confidence_notes) + ".",
            icon="⚠️",
        )

    tab_diag, tab_evidence, tab_context = st.tabs(
        ["🩺 Diagnosis", "📋 Evidence & Files", "🔎 Retrieved Context"]
    )

    # --- Diagnosis tab ------------------------------------------------------
    with tab_diag:
        m1, m2, m3 = st.columns(3)
        m1.metric("Failure type", analysis.failure_type)
        m2.metric("Confidence", f"{analysis.confidence:.0%}")
        m3.metric(
            "Level",
            f"{CONFIDENCE_ICON.get(analysis.confidence_level, '⚪')} "
            f"{analysis.confidence_level.title()}",
        )
        st.progress(
            analysis.confidence, text=f"Confidence: {analysis.confidence:.0%}"
        )
        if analysis.confidence_rationale:
            st.caption(f"Why this confidence: {analysis.confidence_rationale}")

        st.markdown("#### Root cause")
        st.info(analysis.root_cause, icon="🎯")
        if primary is not None:
            st.caption(
                f"Primary suspect: log line {primary.line_number} "
                f"({primary.kind}) — {primary.text[:140]}"
            )

        if analysis.suspect_changes:
            st.markdown("#### Likely related changes")
            st.caption("Correlated with the failure — not proven causation.")
            for i, suspect in enumerate(analysis.suspect_changes, 1):
                st.markdown(f"{i}. **{suspect.change}** — {suspect.why}")

        if analysis.alternative_causes:
            st.markdown("#### Other possible causes")
            for i, alt in enumerate(analysis.alternative_causes, 1):
                st.markdown(f"{i}. {alt.summary}")
                for ev in alt.supporting_evidence:
                    st.caption(f"· {ev}")

        st.markdown("#### Recommended fixes")
        for i, fix in enumerate(analysis.recommended_fixes, 1):
            st.markdown(
                f'<div class="fix-card"><span class="fix-num">{i}</span>'
                f"{fix}</div>",
                unsafe_allow_html=True,
            )

    # --- Evidence tab --------------------------------------------------------
    with tab_evidence:
        if changes:
            st.markdown("#### Likely related changes")
            top = [l for l in change_links if l.score > 0]
            if top:
                for link in top:
                    reasons = "\n".join(f"  - {r}" for r in link.reasons)
                    st.code(
                        f"{link.change.path} "
                        f"({link.change.status}/{link.change.category}) · "
                        f"{link.tier} link · score {link.score:g}\n{reasons}",
                        language="text",
                    )
            else:
                st.caption(
                    "No changed file correlates with the failure log — "
                    "the cause likely lies outside the recent changes."
                )

        st.markdown("#### Error events detected")
        if events:
            for ev in events:
                st.code(
                    f"L{ev.line_number} [{ev.kind}] {ev.text}",
                    language="text",
                )
        else:
            st.caption("No error lines detected in the log.")

        st.markdown("#### Supporting evidence")
        if evidence_items:
            for item in evidence_items:
                badge = "✅" if item.verified else "⚠️ unverified"
                ref = item.reference or "unverified claim"
                st.code(
                    f"{badge} {ref}\n{item.excerpt}",
                    language="text",
                )
        else:
            st.caption("None provided by the model.")

        st.markdown("#### Relevant files")
        if analysis.relevant_files:
            pills = "".join(
                f'<span class="file-pill">{p}</span>'
                for p in analysis.relevant_files
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("None identified.")

    # --- Retrieved context tab ------------------------------------------------
    with tab_context:
        if not retrieved:
            st.caption(
                "No project context was provided — the diagnosis is based "
                "on the log alone."
            )
        for doc in retrieved:
            path = doc.metadata.get("path", "unknown")
            source = doc.metadata.get("source", "")
            line_start = doc.metadata.get("line_start")
            line_end = doc.metadata.get("line_end")
            lines = f" · lines {line_start}-{line_end}" if line_start else ""
            with st.expander(f"`{path}` · {source}{lines}"):
                st.code(doc.page_content[:1500], language="text")


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
inject_css()
render_sidebar()
render_header()

section(1, "Project context", "All inputs are optional — combine them freely.")
github_url = st.text_input(
    "GitHub repository URL",
    placeholder="https://github.com/owner/repo",
)
col_zip, col_files = st.columns(2)
with col_zip:
    zip_file = st.file_uploader("Upload a ZIP archive", type=["zip"])
with col_files:
    individual_files = st.file_uploader(
        "Upload individual files", accept_multiple_files=True
    )

st.markdown("**Recent changes** — optional: what changed before this run?")
diff_text = st.text_area(
    "Paste git diff / git show output",
    height=140,
    placeholder="git diff main...feature   ·   git show HEAD",
)
col_diff, col_cmp = st.columns(2)
with col_diff:
    diff_file = st.file_uploader(
        "…or upload a diff/patch file", type=["diff", "patch", "txt"]
    )
with col_cmp:
    compare_url = st.text_input(
        "…or a GitHub compare URL",
        placeholder="https://github.com/owner/repo/compare/v1.0...v1.1",
    )

section(2, "Failed pipeline log", "Paste the log or upload it as a file.")
log_text = st.text_area(
    "Paste the log output",
    height=200,
    placeholder="[10:00:02] npm ci\n[10:00:10] npm ERR! ERESOLVE unable to "
    "resolve dependency tree\n[10:00:11] Process completed with exit code 1.",
    label_visibility="collapsed",
)
log_file = st.file_uploader("…or upload a log file", type=["log", "txt"])

section(3, "Investigation focus", "Optional — steer the diagnosis.")
user_question = st.text_area(
    "What would you like to investigate?",
    height=90,
    placeholder=(
        "e.g. Is this a dependency conflict?\n"
        "e.g. Did a recent code change cause this failure?\n"
        "e.g. What is the safest fix to apply first?"
    ),
    label_visibility="collapsed",
)

st.markdown("")
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyze_clicked = st.button(
        "🚀 Analyze Failure", type="primary", use_container_width=True
    )

if analyze_clicked:
    error_log = log_text.strip()
    if not error_log and log_file is not None:
        error_log = (
            log_file.getvalue().decode("utf-8", errors="replace").strip()
        )

    if not error_log:
        st.error("Please paste or upload a CI/CD error log.", icon="🚫")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "Please provide a Google Gemini API key in the sidebar.",
            icon="🔑",
        )
    else:
        documents, warnings = gather_documents(
            github_url, zip_file, individual_files
        )
        change_files, change_docs, change_warnings = gather_changes(
            diff_text, diff_file, compare_url
        )
        documents.extend(change_docs)
        warnings.extend(change_warnings)
        for w in warnings:
            st.warning(w, icon="⚠️")

        try:
            result = run_analysis(
                error_log, user_question.strip(), documents, change_files
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}", icon="💥")
        else:
            if result.get("analysis") is None:
                st.error("The model did not return an analysis.")
            else:
                st.session_state["result"] = result
                st.session_state["doc_count"] = len(documents)

# Results persist in session state — visible until a new analysis runs.
if "result" in st.session_state:
    render_results(
        st.session_state["result"],
        st.session_state.get("doc_count", 0),
    )
