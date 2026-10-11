"""Loaders for user-uploaded content: ZIP archives and individual files."""
import tempfile
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.common import extract_zip_to_dir, load_directory

MAX_UPLOAD_FILE_BYTES = 512 * 1024


def load_zip_bytes(zip_bytes: bytes) -> list[Document]:
    """Extract an uploaded ZIP archive and index its contents."""
    with tempfile.TemporaryDirectory(prefix="cicd_zip_") as tmp:
        repo_root = extract_zip_to_dir(zip_bytes, Path(tmp))
        return load_directory(repo_root, source_label="zip")


def load_uploaded_files(files: list[tuple[str, bytes]]) -> list[Document]:
    """Index individually uploaded files.

    Args:
        files: list of (filename, raw_bytes) pairs, e.g. from
               ``st.file_uploader`` items.
    """
    documents: list[Document] = []
    for filename, data in files:
        if len(data) > MAX_UPLOAD_FILE_BYTES:
            continue
        try:
            content = data.decode("utf-8", errors="replace")
        except Exception:
            continue
        if not content.strip():
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={"path": filename, "source": "upload"},
            )
        )
    return documents
