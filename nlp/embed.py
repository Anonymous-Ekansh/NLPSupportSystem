"""Embedding module with caching, L2 normalisation, and bake-off support.

Supports any 384-d model available in fastembed (ONNX, CPU).
Caches embeddings as .npy files keyed by model name + file content hash.

Usage:
    from nlp.embed import embed, EMBED_MODEL
    vecs = embed(texts)                         # (N, 384) float32, L2-normed
    vecs = embed(texts, model="sentence-transformers/all-MiniLM-L6-v2")
"""

import hashlib
import os
import pathlib
from typing import Optional

# Thread caps must be set BEFORE fastembed/onnxruntime initialise: unbounded
# ONNX/OpenMP threads multiply RAM via per-thread workspaces and malloc
# arenas, which OOM-kills the service on 512 MB free-tier hosts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("FASTEMBED_THREADS", "1")

import numpy as np
from fastembed import TextEmbedding

CACHE_DIR = "outputs/cache"

# Supported 384-d models (discovered from fastembed at import time)
SUPPORTED_384: list[str] = []
try:
    for m in TextEmbedding.list_supported_models():
        if m.get("dim") == 384:
            SUPPORTED_384.append(m["model"])
except Exception:
    pass

# Default model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Singleton cache for loaded TextEmbedding instances
_instances: dict[str, TextEmbedding] = {}


def _get_embedder(model: str) -> TextEmbedding:
    """Lazy-load a TextEmbedding singleton."""
    if model not in _instances:
        if model not in SUPPORTED_384:
            supported = "\n  ".join(SUPPORTED_384)
            raise ValueError(
                f"Model '{model}' is not a supported 384-d fastembed model.\n"
                f"Supported:\n  {supported}"
            )
        _instances[model] = TextEmbedding(
            model_name=model,
            threads=int(os.environ.get("FASTEMBED_THREADS", "1")),
            providers=["CPUExecutionProvider"],
        )
    return _instances[model]


def _cache_key(texts: list[str], model: str) -> str:
    """Deterministic cache key from content hash + model name."""
    h = hashlib.sha256()
    h.update(model.encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


def embed(
    texts: list[str],
    model: str = EMBED_MODEL,
    batch_size: int = 256,
    cache_tag: Optional[str] = None,
    force: bool = False,
) -> np.ndarray:
    """Embed texts → (N, 384) float32, L2-normalised.

    Args:
        texts:      list of strings to embed.
        model:      fastembed model name (must be 384-d).
        batch_size: inference batch size.
        cache_tag:  optional human-readable tag prepended to cache filename.
        force:      if True, ignore cache and re-embed.

    Returns:
        np.ndarray of shape (N, 384), dtype float32, each row L2-normalised.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Build cache path
    key = _cache_key(texts, model)
    model_slug = model.replace("/", "__")
    tag = f"{cache_tag}_" if cache_tag else ""
    cache_path = os.path.join(CACHE_DIR, f"{tag}{model_slug}_{key}.npy")

    if not force and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            return arr

    # Compute embeddings
    emb = _get_embedder(model)
    vecs = list(emb.embed(texts, batch_size=batch_size))
    arr = np.array(vecs, dtype=np.float32)

    # L2 normalise
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    arr = arr / norms

    np.save(cache_path, arr)
    return arr


def embed_split(
    split: str,
    model: str = EMBED_MODEL,
    batch_size: int = 256,
    force: bool = False,
) -> np.ndarray:
    """Embed all instructions from a split CSV.  Cached by split name + model."""
    import pandas as pd
    from nlp.normalize import normalize

    df = pd.read_csv(f"outputs/splits/{split}.csv")
    texts = df["instruction"].apply(normalize).tolist()
    return embed(texts, model=model, batch_size=batch_size,
                 cache_tag=split, force=force)


if __name__ == "__main__":
    print(f"Supported 384-d models: {SUPPORTED_384}")
    print(f"Default: {EMBED_MODEL}")

    # Embed all splits with default model
    for split in ("train", "dev", "test"):
        vecs = embed_split(split)
        print(f"  {split}: {vecs.shape}, norm check: {np.linalg.norm(vecs[0]):.6f}")
