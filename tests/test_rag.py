"""Tests for chunking and retrieval — uses HashEmbeddings from conftest
so no real embedding model is downloaded."""
from langchain_core.documents import Document

from src.rag.indexer import build_vectorstore, chunk_documents, retrieve


def test_chunk_documents_adds_path_prefix():
    docs = [
        Document(
            page_content="x = 1\n" * 500,
            metadata={"path": "src/app.py", "source": "upload"},
        )
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 1
    assert all(c.page_content.startswith("# File: src/app.py") for c in chunks)


def test_chunk_documents_respects_chunk_size():
    docs = [Document(page_content="a" * 10_000, metadata={"path": "big.py"})]
    chunks = chunk_documents(docs)
    # every chunk content (minus the "# File:" header line) within bounds
    for c in chunks:
        body = c.page_content.split("\n", 1)[1]
        assert len(body) <= 1200 + 10  # chunk_size + small slack
    assert len(chunks) > 1


def test_chunk_documents_adds_line_ranges():
    """Each chunk knows which line range of the source file it covers."""
    content = "\n".join(f"line {i} " + "x" * 30 for i in range(1, 201))
    docs = [Document(page_content=content, metadata={"path": "big.py"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1

    source_lines = content.splitlines()
    starts = [c.metadata["line_start"] for c in chunks]
    assert starts == sorted(starts)  # chunks walk the file in order
    prev_end = 0
    for c in chunks:
        ls, le = c.metadata["line_start"], c.metadata["line_end"]
        body = c.page_content.split("\n", 1)[1]
        assert le - ls == body.count("\n")
        # line_start points at (or inside) the real line the chunk begins on
        assert source_lines[ls - 1].endswith(body.splitlines()[0])
        assert ls <= prev_end + 1  # no unreported gaps between chunks
        prev_end = le
    assert chunks[0].metadata["line_start"] == 1
    assert chunks[-1].metadata["line_end"] == 200


# --- Vector store + retrieval (fake embeddings) ------------------------------

def test_build_vectorstore_and_retrieve_relevant(
    fake_embeddings, sample_docs
):
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)
    results = retrieve(store, ["npm ci dependencies react"], top_k=2)

    assert len(results) > 0
    paths = {d.metadata["path"] for d in results}
    # The npm-related docs should outrank the unrelated python file.
    assert paths & {"package.json", ".github/workflows/ci.yml"}


def test_retrieve_dedupes_across_queries(fake_embeddings, sample_docs):
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)
    results = retrieve(
        store,
        ["npm ci dependencies", "npm ci dependencies"],  # duplicate query
        top_k=3,
    )
    keys = [
        (d.metadata["path"], hash(d.page_content)) for d in results
    ]
    assert len(keys) == len(set(keys))


def test_retrieve_skips_blank_queries(fake_embeddings, sample_docs):
    store = build_vectorstore(sample_docs, embedding=fake_embeddings)
    assert retrieve(store, ["", "   "], top_k=2) == []


def test_build_vectorstore_empty_docs(fake_embeddings):
    # Chroma raises on empty input — documents the expected behavior.
    import pytest

    with pytest.raises(Exception):
        build_vectorstore([], embedding=fake_embeddings)
