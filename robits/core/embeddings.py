"""Embedding generation via OpenAI-compatible API for semantic memory search."""
from __future__ import annotations

from typing import List


def _embedding_config():
    """Return (base_url, api_key, model) from the shared Config singleton."""
    from robits.core.config import _config as _m
    return (
        getattr(_m, "embedding_base_url", None),
        getattr(_m, "embedding_api_key", None) or _m.client.api_key,
        getattr(_m, "embedding_model", ""),
    )


def get_embedding(text: str, model: str | None = None) -> List[float] | None:
    """Return a float vector for text using the configured embedding model.

    Returns None if no model is configured or the request fails.
    """
    from openai import OpenAI
    base_url, api_key, configured_model = _embedding_config()
    effective_model = model or configured_model
    if not effective_model:
        return None
    try:
        kwargs: dict = {"api_key": api_key or "not-needed"}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        response = client.embeddings.create(input=[text], model=effective_model)
        return response.data[0].embedding
    except Exception:
        return None


def embedding_model_name() -> str:
    """Return the configured embedding model name, or empty string if none."""
    from robits.core.config import _config as _m
    return getattr(_m, "embedding_model", "")
