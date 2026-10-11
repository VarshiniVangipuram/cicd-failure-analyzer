"""Thin wrapper around the Gemini chat model.

Centralizing construction keeps the model name / key handling in one place,
so nodes and the UI don't duplicate it.
"""
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings


def get_llm(api_key: str | None = None, model: str | None = None) -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model.

    Args:
        api_key: Optional override; falls back to GOOGLE_API_KEY env var.
        model: Optional override; falls back to GEMINI_MODEL env var.
    """
    key = api_key or settings.google_api_key
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to a .env file or pass it "
            "via the Streamlit sidebar."
        )
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_model,
        google_api_key=key,
        temperature=0.2,
    )
