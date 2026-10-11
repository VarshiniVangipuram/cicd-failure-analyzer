# CI/CD Failure Analyzer

An AI-powered debugging assistant that analyzes failed CI/CD pipeline logs,
retrieves relevant project context via RAG, and produces a structured
root-cause diagnosis with evidence, relevant files, and recommended fixes.

## Tech stack

- **LangChain + LangGraph** — analysis pipeline orchestration
- **ChromaDB + sentence-transformers** — local embeddings & retrieval (no API key)
- **Google Gemini** — structured failure diagnosis
- **Streamlit** — UI

## How it works

```
uploads (GitHub URL / ZIP / files) ──► Documents ──► chunks ──► ChromaDB
                                                                   │
error log ──► parse_log (regex) ──► retrieve_context ──► analyze_failure (Gemini)
                                        ▲                           │
                                        └──────── queries ──────────┘
                                                                   ▼
                                          FailureAnalysis { type, root_cause,
                                          evidence, files, fixes, confidence }
```

- `parse_log` — extracts error signatures from the log deterministically.
- `retrieve_context` — queries the ephemeral Chroma index with those signatures.
- `analyze_failure` — Gemini produces a typed `FailureAnalysis` via structured output.

The vector store is **in-memory and rebuilt per run**, so context never
leaks between analyses.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env                            # add your GOOGLE_API_KEY
```

Get a Gemini key at https://aistudio.google.com/app/apikey

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```

## Project structure

```
app.py                  # Streamlit UI
src/
  config.py             # env-driven settings
  schemas.py            # FailureAnalysis pydantic model
  llm.py                # Gemini wrapper
  ingestion/            # github.py · uploads.py · common.py
  rag/                  # embeddings.py · indexer.py
  graph/                # state.py · nodes.py · builder.py
tests/                  # unit tests (no API key required)
```

## Configuration (env vars)

| Variable          | Default                                   |
|-------------------|-------------------------------------------|
| `GOOGLE_API_KEY`  | **required**                              |
| `GEMINI_MODEL`    | `gemini-3.8-flash`                        |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2`  |
| `CHUNK_SIZE`      | `1200`                                    |
| `CHUNK_OVERLAP`   | `200`                                     |
| `TOP_K`           | `6`                                       |
