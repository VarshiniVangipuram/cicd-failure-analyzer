"""Chunking + in-memory ChromaDB index, built fresh per analysis run."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.rag.embeddings import get_embedding_model


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Each chunk's metadata carries the source ``path`` plus a
    ``line_start``/``line_end`` range so retrieved context can be cited
    back to precise locations, e.g. ``src/app.py:34-40``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks: list[Document] = []
    for doc in documents:
        for chunk in splitter.split_documents([doc]):
            start = chunk.metadata.get("start_index")
            if start is None or start < 0:
                # Older splitters may omit start_index — locate manually.
                start = doc.page_content.find(chunk.page_content)
            if start < 0:
                start = 0
            chunk.metadata["line_start"] = doc.page_content.count("\n", 0, start) + 1
            chunk.metadata["line_end"] = (
                chunk.metadata["line_start"] + chunk.page_content.count("\n")
            )
            path = chunk.metadata.get("path", "unknown")
            # Prepend the path so the embedding itself is path-aware.
            chunk.page_content = f"# File: {path}\n{chunk.page_content}"
            chunks.append(chunk)
    return chunks


def build_vectorstore(
    documents: list[Document], embedding=None
) -> Chroma:
    """Build an ephemeral (in-memory) Chroma index from documents.

    Ephemeral by design: every analysis run indexes only the context the
    user just provided, so results never leak across runs.

    Args:
        embedding: optional override — tests inject a fake embedding here
                   to avoid downloading the real model.
    """
    chunks = chunk_documents(documents)
    return Chroma.from_documents(
        chunks, embedding=embedding or get_embedding_model()
    )


def retrieve(
    store: Chroma, queries: list[str], top_k: int | None = None
) -> list[Document]:
    """Run each query against the store and return deduplicated results."""
    k = top_k or settings.top_k
    retriever = store.as_retriever(search_kwargs={"k": k})
    seen: set[str] = set()
    results: list[Document] = []
    for query in queries:
        if not query.strip():
            continue
        for doc in retriever.invoke(query):
            key = f"{doc.metadata.get('path')}::{hash(doc.page_content)}"
            if key not in seen:
                seen.add(key)
                results.append(doc)
    return results
