#!/usr/bin/env python3
"""Build 2D map coordinates for TRAIN embeddings.

Uses umap-learn if available; falls back to sklearn TSNE on a
stratified 6000-point sample.  Saves:

    outputs/map_points.json   — stratified sample (≤ 3000 pts) with
                                 {x, y, intent, category, index}

Also prints a summary of the projection.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from nlp.embed import embed_split


SEED = 42
MAP_OUT = "outputs/map_points.json"
MAX_SAMPLE = 3000    # points saved to map_points.json
UMAP_SAMPLE = 6000   # max points for TSNE fallback


def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Return a stratified sample of ≤ n rows, balanced by intent."""
    per_class = max(1, n // df["intent"].nunique())
    return (
        df.groupby("intent", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
          .sample(frac=1, random_state=seed)
          .head(n)
          .sort_index()
    )


def project_2d(embs: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D using UMAP or TSNE fallback."""
    try:
        import umap
        print("  Using UMAP …")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            metric="cosine", random_state=SEED,
        )
        return reducer.fit_transform(embs)
    except ImportError:
        print("  umap-learn not available, falling back to TSNE …")
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2, perplexity=30, learning_rate="auto",
            init="pca", random_state=SEED,
        ).fit_transform(embs)


def main():
    print("=" * 60)
    print("BUILD MAP COORDINATES")
    print("=" * 60)

    train_df = pd.read_csv("outputs/splits/train.csv")
    train_emb = embed_split("train")
    n_total = len(train_df)
    print(f"  TRAIN: {n_total} rows, embeddings: {train_emb.shape}")

    # For TSNE fallback, we may need to subsample
    use_umap = True
    try:
        import umap  # noqa: F401
    except ImportError:
        use_umap = False
        print(f"  TSNE fallback: sampling {UMAP_SAMPLE} points …")

    if use_umap:
        # UMAP can handle full dataset
        proj_embs = train_emb
        proj_indices = np.arange(n_total)
    else:
        # Stratified sample for TSNE
        sample_df = stratified_sample(train_df, UMAP_SAMPLE)
        proj_indices = sample_df.index.values
        proj_embs = train_emb[proj_indices]

    print(f"  Projecting {len(proj_embs)} points to 2D …")
    coords_2d = project_2d(proj_embs)
    print(f"  Projection done: {coords_2d.shape}")

    # Build full coordinate map (indexed by original row)
    all_x = np.full(n_total, np.nan, dtype=np.float32)
    all_y = np.full(n_total, np.nan, dtype=np.float32)
    all_x[proj_indices] = coords_2d[:, 0]
    all_y[proj_indices] = coords_2d[:, 1]

    # Save stratified sample ≤ 3000 for JSON
    save_df = stratified_sample(train_df, MAX_SAMPLE)
    save_indices = save_df.index.values

    coordinates = []
    for idx in save_indices:
        x_val = float(all_x[idx]) if not np.isnan(all_x[idx]) else None
        y_val = float(all_y[idx]) if not np.isnan(all_y[idx]) else None
        row = train_df.iloc[idx]
        coordinates.append({
            "index": int(idx),
            "x": round(x_val, 4) if x_val is not None else None,
            "y": round(y_val, 4) if y_val is not None else None,
            "intent": str(row["intent"]),
            "category": str(row["category"]),
        })

    output = {
        "n_total": n_total,
        "n_projected": int(len(proj_indices)),
        "n_saved": len(coordinates),
        "method": "UMAP" if use_umap else "TSNE",
        "coordinates": coordinates,
    }

    os.makedirs(os.path.dirname(MAP_OUT), exist_ok=True)
    with open(MAP_OUT, "w") as f:
        json.dump(output, f, indent=1)

    print(f"\n  Saved {len(coordinates)} points to {MAP_OUT}")
    n_with_coords = sum(1 for c in coordinates if c["x"] is not None)
    print(f"  Points with coordinates: {n_with_coords} / {len(coordinates)}")
    print("  Done.")


if __name__ == "__main__":
    main()
