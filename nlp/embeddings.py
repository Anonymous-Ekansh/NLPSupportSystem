"""Thin wrapper around fastembed for BAAI/bge-small-en-v1.5 (384-d).

Usage:
    from nlp.embeddings import get_embedder, embed_texts
"""

import numpy as np
from fastembed import TextEmbedding

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_INSTANCE = None


def get_embedder() -> TextEmbedding:
    """Lazy-load singleton embedder."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = TextEmbedding(model_name=_MODEL_NAME)
    return _INSTANCE


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed a list of strings, return (N, 384) float32 array."""
    emb = get_embedder()
    vecs = list(emb.embed(texts, batch_size=batch_size))
    return np.array(vecs, dtype=np.float32)
