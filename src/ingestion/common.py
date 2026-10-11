"""Shared helpers for turning files on disk into LangChain Documents."""
import os
from pathlib import Path

from langchain_core.documents import Document

# Extensions we treat as indexable text (code, config, docs, CI files).
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".ps1", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".json", ".xml", ".md", ".txt", ".rst", ".dockerfile", ".tf",
    ".properties", ".env.example", ".gradle", ".lock", ".sql", ".html",
    ".css", ".scss", ".vue", ".svelte",
}

# Filenames with no extension that are still worth indexing.
SPECIAL_FILENAMES = {
    "dockerfile", "makefile", "jenkinsfile", "vagrantfile", "procfile",
    "gemfile", "rakefile", ".gitignore", ".dockerignore", ".env.example",
}

# Directories that are never useful for failure analysis.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", ".idea", ".vscode", ".next", ".cache",
    "coverage", ".pytest_cache", ".mypy_cache", "vendor", "bin", "obj",
}

MAX_FILE_BYTES = 512 * 1024  # skip files larger than 512 KB


def is_text_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in TEXT_EXTENSIONS or name in SPECIAL_FILENAMES


def load_directory(root: str | Path, source_label: str = "local") -> list[Document]:
    """Walk a directory and load every indexable text file as a Document.

    Each Document's metadata carries a repo-relative ``path`` plus a
    ``source`` label ("github", "zip", "upload") so the UI can show where
    a piece of evidence came from.
    """
    root = Path(root)
    documents: list[Document] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if not is_text_file(file_path):
                continue
            try:
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if not content.strip():
                continue

            rel_path = file_path.relative_to(root).as_posix()
            documents.append(
                Document(
                    page_content=content,
                    metadata={"path": rel_path, "source": source_label},
                )
            )

    return documents


def extract_zip_to_dir(zip_bytes: bytes, dest_dir: Path) -> Path:
    """Extract zip bytes into dest_dir and return the effective repo root.

    GitHub archives contain a single top-level ``<repo>-<branch>/`` folder;
    we unwrap that so relative paths look clean.
    """
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Basic zip-slip protection.
        for member in zf.namelist():
            target = (dest_dir / member).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise ValueError(f"Unsafe path in zip archive: {member}")
        zf.extractall(dest_dir)

    children = [p for p in dest_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest_dir
