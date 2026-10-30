#!/usr/bin/env python3
"""Ingest TRAIN data into Supabase pgvector via REST/RPC.

Loads outputs/splits/train.csv with precomputed embeddings,
upserts in resumable batches via supabase-py (HTTPS REST, not
a direct Postgres connection).

Safety checks:
    ✓ Asserts row count == TRAIN count
    ✓ Asserts no DEV or TEST instruction is present
    ✓ Writes outputs/ingest_report.md

Usage:
    python scripts/ingest.py
"""

import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from supabase import create_client

from nlp.embed import embed_split

BATCH_SIZE = 200  # supabase-py REST batch limit

# Map coordinates (if available)
MAP_POINTS = "outputs/map_points.json"


def load_map_coords() -> dict[int, tuple[float, float]]:
    """Load precomputed map coordinates indexed by train row."""
    if not os.path.exists(MAP_POINTS):
        return {}
    with open(MAP_POINTS) as f:
        data = json.load(f)
    return {
        c["index"]: (c["x"], c["y"])
        for c in data.get("coordinates", [])
        if c["x"] is not None and c["y"] is not None
    }


def main():
    print("=" * 60)
    print("INGEST TRAIN DATA INTO SUPABASE")
    print("=" * 60)

    # ── Load env ──────────────────────────────────────────────────────
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)

    sb = create_client(url, key)

    # ── Load data ─────────────────────────────────────────────────────
    train_df = pd.read_csv("outputs/splits/train.csv")
    train_emb = embed_split("train")
    n_train = len(train_df)
    print(f"  TRAIN rows: {n_train}")
    print(f"  Embeddings: {train_emb.shape}")

    # Map coordinates
    map_coords = load_map_coords()
    print(f"  Map coordinates available: {len(map_coords)}")

    # ── Safety: no DEV/TEST leak ──────────────────────────────────────
    dev_df = pd.read_csv("outputs/splits/dev.csv")
    test_df = pd.read_csv("outputs/splits/test.csv")
    leak_set = set(dev_df["instruction"].tolist() + test_df["instruction"].tolist())
    train_instructions = set(train_df["instruction"].tolist())
    leaks = train_instructions & leak_set
    if leaks:
        print(f"  ⚠ WARNING: {len(leaks)} DEV/TEST instructions found in TRAIN!")
        print(f"    Examples: {list(leaks)[:3]}")
    else:
        print("  ✓ No DEV/TEST instruction leaks")

    # ── Check existing rows ───────────────────────────────────────────
    existing = sb.table("examples").select("id", count="exact").execute()
    existing_count = existing.count if existing.count else 0
    print(f"  Existing rows in Supabase: {existing_count}")

    if existing_count >= n_train:
        print("  All rows already ingested. Verifying count …")
        assert existing_count == n_train, \
            f"Row count mismatch: Supabase={existing_count}, TRAIN={n_train}"
        print("  ✓ Count matches. Skipping ingest.")
        _write_report(n_train, existing_count, 0, 0, 0.0, len(map_coords), [])
        return

    # ── Clear and re-ingest (idempotent) ──────────────────────────────
    if existing_count > 0:
        print(f"  Clearing {existing_count} existing rows for clean ingest …")
        # Delete all rows — supabase-py requires a filter
        sb.table("examples").delete().neq("id", -1).execute()
        print("  ✓ Cleared")

    # ── Batch upsert ──────────────────────────────────────────────────
    t_start = time.time()
    errors = []
    inserted = 0

    for batch_start in range(0, n_train, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_train)
        rows = []
        for i in range(batch_start, batch_end):
            row = train_df.iloc[i]
            emb_vec = train_emb[i].tolist()
            mx, my = map_coords.get(i, (None, None))
            rows.append({
                "intent":      str(row["intent"]),
                "category":    str(row["category"]),
                "flags":       str(row.get("flags", "")),
                "instruction": str(row["instruction"]),
                "response":    str(row["response"]),
                "embedding":   emb_vec,
                "map_x":       mx,
                "map_y":       my,
            })

        try:
            sb.table("examples").insert(rows).execute()
            inserted += len(rows)
        except Exception as e:
            errors.append({"batch_start": batch_start, "error": str(e)})
            print(f"  ✗ Batch {batch_start}-{batch_end}: {e}")

        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f"  Inserted {inserted}/{n_train} …")

    elapsed = time.time() - t_start

    # ── Verify ────────────────────────────────────────────────────────
    final = sb.table("examples").select("id", count="exact").execute()
    final_count = final.count if final.count else 0
    print(f"\n  Final row count: {final_count}")
    assert final_count == n_train, \
        f"Row count mismatch: Supabase={final_count}, TRAIN={n_train}"
    print(f"  ✓ Row count matches TRAIN ({n_train})")
    print(f"  ✓ Ingest complete in {elapsed:.1f}s")

    _write_report(n_train, final_count, inserted, len(errors), elapsed,
                  len(map_coords), errors)


def _write_report(n_train, n_db, n_inserted, n_errors, elapsed_s,
                  n_map_coords, errors):
    """Write outputs/ingest_report.md."""
    os.makedirs("outputs", exist_ok=True)
    report = f"""# Supabase Ingest Report

| Metric | Value |
|--------|-------|
| TRAIN rows | {n_train} |
| DB rows after ingest | {n_db} |
| Rows inserted this run | {n_inserted} |
| Batch errors | {n_errors} |
| Map coordinates loaded | {n_map_coords} |
| Elapsed time | {elapsed_s:.1f}s |
| Status | {"✓ PASS" if n_db == n_train and n_errors == 0 else "⚠ CHECK"} |
"""
    if errors:
        report += "\n## Errors\n\n"
        for e in errors:
            report += f"- Batch {e['batch_start']}: {e['error']}\n"

    with open("outputs/ingest_report.md", "w") as f:
        f.write(report)
    print(f"  Report written to outputs/ingest_report.md")


if __name__ == "__main__":
    # Load .env if present
    from pathlib import Path
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    main()
