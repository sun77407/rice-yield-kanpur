"""Embeddings through Ollama's ``nomic-embed-text`` (ADR-008), with its task prefixes."""

from __future__ import annotations

import os

import numpy as np
import ollama

EMBED_MODEL = os.environ.get("EMBED_MODEL") or "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
_client: ollama.Client | None = None


def client() -> ollama.Client:
    """Return the shared Ollama client (host from ``OLLAMA_HOST`` if set)."""
    global _client
    if _client is None:
        _client = ollama.Client(host=os.environ.get("OLLAMA_HOST") or None, timeout=300)
    return _client


def embed(texts: list[str], prefix: str = DOC_PREFIX, batch: int = 32) -> np.ndarray:
    """Return L2-normalised embeddings, shape ``(len(texts), dim)``."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client().embed(model=EMBED_MODEL, input=[prefix + t for t in texts[i : i + batch]])
        out.extend(resp["embeddings"])
    arr = np.asarray(out, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
