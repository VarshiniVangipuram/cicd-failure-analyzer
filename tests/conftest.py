"""Shared fixtures: fake embeddings + fake LLM so tests need no network
or API keys."""
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from src.schemas import FailureAnalysis


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedder.

    Semantically crude but good enough for tests: documents and queries
    sharing tokens end up with overlapping non-zero vector entries.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            vec[hash(token) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Stand-in for ChatGoogleGenerativeAI.

    ``with_structured_output`` returns a runnable that yields ``analysis``
    on invoke — or raises ``exc`` to simulate a Gemini API failure.
    """

    def __init__(
        self,
        analysis: FailureAnalysis | None = None,
        exc: Exception | None = None,
    ):
        self.analysis = analysis or FailureAnalysis(
            failure_type="dependency_error",
            root_cause="npm cannot resolve the dependency tree",
            evidence=["npm ERR! ERESOLVE unable to resolve dependency tree"],
            relevant_files=["package.json"],
            recommended_fixes=["Pin dependency versions", "Re-run npm ci"],
            confidence=0.85,
            confidence_level="high",
        )
        self.exc = exc

    def with_structured_output(self, schema):
        def _run(_prompt_value):
            if self.exc:
                raise self.exc
            return self.analysis

        return RunnableLambda(_run)


@pytest.fixture
def fake_embeddings() -> HashEmbeddings:
    return HashEmbeddings()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_docs():
    from langchain_core.documents import Document

    return [
        Document(
            page_content='{"dependencies": {"react": "^18.0.0"}}',
            metadata={"path": "package.json", "source": "upload"},
        ),
        Document(
            page_content="name: ci\nsteps:\n  - run: npm ci\n  - run: npm test",
            metadata={"path": ".github/workflows/ci.yml", "source": "upload"},
        ),
        Document(
            page_content="def main():\n    print('hello')",
            metadata={"path": "src/app.py", "source": "upload"},
        ),
    ]


@pytest.fixture
def sample_log() -> str:
    return (
        "[10:00:01] Installing dependencies...\n"
        "[10:00:02] npm ci\n"
        "[10:00:10] npm ERR! code ERESOLVE\n"
        "[10:00:10] npm ERR! ERESOLVE unable to resolve dependency tree\n"
        "[10:00:11] Process completed with exit code 1.\n"
    )
