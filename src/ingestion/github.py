"""Load a public GitHub repository by downloading its zip archive.

We deliberately avoid ``git clone`` so the app works on machines without
git installed and doesn't need a GitHub token for public repos.
"""
import os
import re
import tempfile
from pathlib import Path

import requests
from langchain_core.documents import Document

from src.changes import FileChange, apply_patch_text, classify_change
from src.ingestion.common import extract_zip_to_dir, load_directory

GITHUB_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)/?$"
)

COMPARE_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)"
    r"/compare/(?P<range>.+?)/?$"
)


def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) from a GitHub URL, or raise ValueError."""
    match = GITHUB_URL_PATTERN.match(url.strip())
    if not match:
        raise ValueError(
            f"Invalid GitHub URL: {url!r}. "
            "Expected format: https://github.com/owner/repo"
        )
    repo = match.group("repo")
    if repo.endswith(".git"):
        repo = repo[:-4]
    return match.group("owner"), repo


def _download_repo_zip(owner: str, repo: str, timeout: int = 60) -> bytes:
    """Try 'main' then 'master' branch archives via codeload."""
    last_error: Exception | None = None
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.content
            last_error = RuntimeError(
                f"HTTP {response.status_code} for branch '{branch}'"
            )
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(
        f"Could not download {owner}/{repo} (tried 'main' and 'master'). "
        f"Last error: {last_error}"
    )


def load_github_repo(url: str) -> list[Document]:
    """Download and index a public GitHub repository."""
    owner, repo = parse_github_url(url)
    zip_bytes = _download_repo_zip(owner, repo)

    with tempfile.TemporaryDirectory(prefix="cicd_repo_") as tmp:
        repo_root = extract_zip_to_dir(zip_bytes, Path(tmp))
        documents = load_directory(repo_root, source_label="github")

    for doc in documents:
        doc.metadata["repo"] = f"{owner}/{repo}"
    return documents


# ---------------------------------------------------------------------------
# Compare two commits via the GitHub API (no local git needed)
# ---------------------------------------------------------------------------

_GITHUB_STATUS_MAP = {
    "added": "added",
    "removed": "deleted",
    "modified": "modified",
    "renamed": "renamed",
    "changed": "modified",
    "copied": "added",
}


def _github_headers() -> dict:
    """Accept header plus an *optional* token from the environment only.

    GITHUB_TOKEN is read here, sent as an Authorization header, and never
    logged or rendered — it also unlocks private repos the token can read.
    """
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_github_compare_url(url: str) -> tuple[str, str, str, str]:
    """Return (owner, repo, base, head) from a GitHub compare URL.

    Ref names may contain slashes (feature/x); the '...' or '..' separator
    is the divider, not '/'.
    """
    match = COMPARE_URL_PATTERN.match(url.strip())
    if not match:
        raise ValueError(
            f"Invalid GitHub compare URL: {url!r}. Expected format: "
            "https://github.com/owner/repo/compare/base...head"
        )
    repo = match.group("repo")
    if repo.endswith(".git"):
        repo = repo[:-4]
    rng = match.group("range")
    sep = "..." if "..." in rng else ".."
    base, _, head = rng.partition(sep)
    if not base or not head:
        raise ValueError(
            f"Invalid compare range {rng!r} — expected base...head."
        )
    return match.group("owner"), repo, base, head


def load_github_compare(url: str, timeout: int = 30) -> list[FileChange]:
    """Fetch a commit comparison and return the changed files.

    Files whose patch GitHub omits (too large, binary) are returned with
    ``patch_incomplete=True`` so correlation can flag the blind spot.
    """
    owner, repo, base, head = parse_github_compare_url(url)
    api = f"https://api.github.com/repos/{owner}/{repo}/compare/{base}...{head}"

    try:
        response = requests.get(api, timeout=timeout, headers=_github_headers())
    except requests.RequestException as exc:
        raise RuntimeError(f"GitHub compare request failed: {exc}") from exc
    if response.status_code != 200:
        hint = ""
        if response.status_code in (401, 403, 404):
            hint = (
                " (set the GITHUB_TOKEN env var with read access for private repos)"
                if not os.getenv("GITHUB_TOKEN", "").strip()
                else " (check that GITHUB_TOKEN has read access to this repository)"
            )
        raise RuntimeError(
            f"GitHub compare returned HTTP {response.status_code}{hint}"
        )

    changes: list[FileChange] = []
    for entry in response.json().get("files", []):
        filename = entry.get("filename", "")
        if not filename:
            continue
        change = FileChange(
            path=filename,
            status=_GITHUB_STATUS_MAP.get(entry.get("status", ""), "modified"),
            old_path=entry.get("previous_filename"),
        )
        patch = entry.get("patch")
        if patch:
            apply_patch_text(change, patch)
        else:
            change.patch_incomplete = True
        change.category = classify_change(change.path or change.old_path or "")
        changes.append(change)
    return changes
