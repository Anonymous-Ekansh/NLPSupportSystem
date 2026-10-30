#!/usr/bin/env python3
"""Parity benchmark: pgvector RPC vs in-memory index.

On 50 TEST queries, compare:
    - top-5 IDs from pgvector match_hybrid vs in-memory hybrid
    - p50 / p95 latency of each backend

Outputs: outputs/evidence/retrieval_backend.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Load .env
from pathlib import Path
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    print("=" * 60)
    print("RETRIEVAL PARITY BENCHMARK")
    print("=" * 60)

    from nlp.a1 import IntentAnalyst
    from nlp.a2 import A2Builder
    from nlp.embed import embed
    from nlp.normalize import normalize

    # Load models
    print("  Loading A1 …")
    a1 = IntentAnalyst()
    print("  Building A2 (in-memory) …")
    a2 = A2Builder().build()

    # Supabase client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_KEY not set.")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(url, key)

    # Verify Supabase has data
    count_resp = sb.table("examples").select("id", count="exact").limit(1).execute()
    n_db = count_resp.count or 0
    print(f"  Supabase examples: {n_db}")
    if n_db == 0:
        print("  ERROR: No data in Supabase. Run scripts/ingest.py first.")
        sys.exit(1)

    # Sample 50 TEST queries
    test_df = pd.read_csv("outputs/splits/test.csv")
    sample = test_df.sample(50, random_state=42)
    queries = sample["instruction"].tolist()
    print(f"  Benchmarking {len(queries)} queries …\n")

    results = []
    pg_latencies = []
    mem_latencies = []
    overlaps = []

    for i, query in enumerate(queries):
        q_norm = normalize(query)
        q_emb = embed([q_norm])[0]
        a1_out = a1.analyze(query)
        intent = a1_out["intent"]

        # ── In-memory ────────────────────────────────────────────────
        t0 = time.perf_counter()
        mem_result = a2.respond(query, a1_out, mode="hybrid", k=5)
        mem_ms = (time.perf_counter() - t0) * 1000
        mem_latencies.append(mem_ms)

        # Get in-memory top-5 instructions (as proxy IDs)
        mem_instructions = set(
            ex["instruction"] for ex in mem_result["supporting_examples"][:5]
        )

        # ── pgvector ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            pg_resp = sb.rpc("match_hybrid", {
                "query_embedding": q_emb.tolist(),
                "query_text": query,
                "match_count": 5,
                "filter_intents": [intent],
            }).execute()
            pg_ms = (time.perf_counter() - t0) * 1000
            pg_latencies.append(pg_ms)

            pg_instructions = set(
                r["instruction"] for r in (pg_resp.data or [])[:5]
            )
        except Exception as e:
            pg_ms = (time.perf_counter() - t0) * 1000
            pg_latencies.append(pg_ms)
            pg_instructions = set()
            print(f"  [{i}] pgvector error: {e}")

        # ── Overlap ──────────────────────────────────────────────────
        if pg_instructions and mem_instructions:
            overlap = len(pg_instructions & mem_instructions) / max(
                len(pg_instructions), len(mem_instructions)
            )
        else:
            overlap = 0.0
        overlaps.append(overlap)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/50  overlap={overlap:.2f}  "
                  f"pg={pg_ms:.0f}ms  mem={mem_ms:.0f}ms")

    # ── Summary ──────────────────────────────────────────────────────
    pg_lat = np.array(pg_latencies)
    mem_lat = np.array(mem_latencies)
    ovl = np.array(overlaps)

    report = {
        "n_queries": len(queries),
        "pgvector": {
            "p50_ms": round(float(np.percentile(pg_lat, 50)), 1),
            "p95_ms": round(float(np.percentile(pg_lat, 95)), 1),
            "mean_ms": round(float(pg_lat.mean()), 1),
        },
        "in_memory": {
            "p50_ms": round(float(np.percentile(mem_lat, 50)), 1),
            "p95_ms": round(float(np.percentile(mem_lat, 95)), 1),
            "mean_ms": round(float(mem_lat.mean()), 1),
        },
        "overlap": {
            "mean": round(float(ovl.mean()), 4),
            "min": round(float(ovl.min()), 4),
            "max": round(float(ovl.max()), 4),
            "median": round(float(np.median(ovl)), 4),
        },
        "note": (
            "Overlap measures fraction of shared top-5 results (by instruction text) "
            "between pgvector hybrid RPC and in-memory hybrid retrieval. "
            "Differences are expected: pgvector uses websearch_to_tsquery + HNSW "
            "while in-memory uses exact TF-IDF + dot-product embeddings with RRF."
        ),
    }

    os.makedirs("outputs/evidence", exist_ok=True)
    out_path = "outputs/evidence/retrieval_backend.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  pgvector:  p50={report['pgvector']['p50_ms']}ms  "
          f"p95={report['pgvector']['p95_ms']}ms")
    print(f"  in-memory: p50={report['in_memory']['p50_ms']}ms  "
          f"p95={report['in_memory']['p95_ms']}ms")
    print(f"  overlap:   mean={report['overlap']['mean']:.2%}  "
          f"min={report['overlap']['min']:.2%}  max={report['overlap']['max']:.2%}")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
