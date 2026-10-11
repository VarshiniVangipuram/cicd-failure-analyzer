"""Tests for the ingestion layer — network calls are mocked."""
import io
import zipfile
from pathlib import Path

import pytest
import requests

import src.ingestion.github as github_mod
from src.ingestion.common import extract_zip_to_dir, load_directory
from src.ingestion.github import (
    _download_repo_zip,
    load_github_repo,
    parse_github_url,
)
from src.ingestion.uploads import load_uploaded_files, load_zip_bytes


def _make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


# --- GitHub URL parsing -----------------------------------------------------

def test_parse_github_url_valid():
    assert parse_github_url("https://github.com/owner/repo") == ("owner", "repo")
    assert parse_github_url("https://github.com/owner/repo/") == ("owner", "repo")
    assert parse_github_url("http://www.github.com/a/b.git") == ("a", "b")
    assert parse_github_url("  https://github.com/a/b  ") == ("a", "b")


def test_parse_github_url_invalid():
    with pytest.raises(ValueError):
        parse_github_url("https://gitlab.com/owner/repo")
    with pytest.raises(ValueError):
        parse_github_url("not a url")
    with pytest.raises(ValueError):
        parse_github_url("")


# --- GitHub download (mocked HTTP) ------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


def test_download_repo_zip_success(monkeypatch):
    payload = _make_zip({"repo-main/a.py": "x"})
    monkeypatch.setattr(
        github_mod.requests, "get",
        lambda url, timeout: _FakeResponse(200, payload),
    )
    assert _download_repo_zip("o", "r") == payload


def test_download_repo_zip_falls_back_to_master(monkeypatch):
    payload = _make_zip({"repo-master/a.py": "x"})
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        if "heads/main" in url:
            return _FakeResponse(404)
        return _FakeResponse(200, payload)

    monkeypatch.setattr(github_mod.requests, "get", fake_get)
    assert _download_repo_zip("o", "r") == payload
    assert any("heads/master" in u for u in calls)


def test_download_repo_zip_repo_not_found(monkeypatch):
    monkeypatch.setattr(
        github_mod.requests, "get",
        lambda url, timeout: _FakeResponse(404),
    )
    with pytest.raises(RuntimeError, match="Could not download"):
        _download_repo_zip("ghost", "missing")


def test_download_repo_zip_network_error(monkeypatch):
    def boom(url, timeout):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(github_mod.requests, "get", boom)
    with pytest.raises(RuntimeError, match="Could not download"):
        _download_repo_zip("o", "r")


def test_load_github_repo_end_to_end(monkeypatch):
    payload = _make_zip(
        {"repo-main/app.py": "print(1)", "repo-main/.github/workflows/ci.yml": "on: push"}
    )
    monkeypatch.setattr(
        github_mod.requests, "get",
        lambda url, timeout: _FakeResponse(200, payload),
    )
    docs = load_github_repo("https://github.com/o/repo")
    paths = {d.metadata["path"] for d in docs}
    assert paths == {"app.py", ".github/workflows/ci.yml"}
    assert all(d.metadata["repo"] == "o/repo" for d in docs)


# --- Directory loading -------------------------------------------------------

def test_load_directory_skips_non_text_and_gitignored(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('hi')")
    (tmp_path / "ci.yml").write_text("name: ci")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "junk.js").write_text("x")

    docs = load_directory(tmp_path)
    paths = {d.metadata["path"] for d in docs}

    assert paths == {"app.py", "ci.yml"}
    assert all(d.metadata["source"] == "local" for d in docs)


def test_load_directory_empty(tmp_path: Path):
    assert load_directory(tmp_path) == []


# --- ZIP extraction ----------------------------------------------------------

def test_extract_zip_unwraps_single_root(tmp_path: Path):
    zip_bytes = _make_zip({"repo-main/app.py": "x", "repo-main/a/b.yml": "y"})
    root = extract_zip_to_dir(zip_bytes, tmp_path)
    assert (root / "app.py").exists()
    assert root.name == "repo-main"


def test_extract_zip_rejects_zip_slip(tmp_path: Path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        extract_zip_to_dir(buf.getvalue(), tmp_path)


def test_load_zip_bytes():
    zip_bytes = _make_zip({"src/main.py": "print(1)", "README.md": "# hi"})
    docs = load_zip_bytes(zip_bytes)
    paths = {d.metadata["path"] for d in docs}
    assert paths == {"src/main.py", "README.md"}
    assert all(d.metadata["source"] == "zip" for d in docs)


def test_load_zip_bytes_corrupt():
    with pytest.raises(zipfile.BadZipFile):
        load_zip_bytes(b"this is not a zip file")


def test_load_zip_bytes_empty():
    with pytest.raises(zipfile.BadZipFile):
        load_zip_bytes(b"")


# --- Individual uploads ------------------------------------------------------

def test_load_uploaded_files():
    files = [
        ("config.yml", b"key: value"),
        ("empty.py", b"   "),
        ("big.bin", b"x" * (513 * 1024)),
    ]
    docs = load_uploaded_files(files)
    assert [d.metadata["path"] for d in docs] == ["config.yml"]


def test_load_uploaded_files_empty_list():
    assert load_uploaded_files([]) == []
