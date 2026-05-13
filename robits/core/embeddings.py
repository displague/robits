"""Embedding generation via OpenAI-compatible API for semantic memory search."""
from __future__ import annotations

import os
from typing import List


def _embedding_client():
    """Return an OpenAI client pointed at the configured embedding base URL."""
    from openai import OpenAI
    base_url = (
        os.environ.get("ROBITS_EMBEDDING_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )
    api_key = os.environ.get("ROBITS_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "not-needed")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_embedding(text: str, model: str | None = None) -> List[float] | None:
    """Return a float vector for text using the configured embedding model.

    Returns None if no model is configured or the request fails.
    """
    effective_model = model or os.environ.get("ROBITS_EMBEDDING_MODEL", "")
    if not effective_model:
        return None
    try:
        client = _embedding_client()
        response = client.embeddings.create(input=[text], model=effective_model)
        return response.data[0].embedding
    except Exception:
        return None


def embedding_model_name() -> str:
    """Return the configured embedding model name, or empty string if none."""
    return os.environ.get("ROBITS_EMBEDDING_MODEL", "")
