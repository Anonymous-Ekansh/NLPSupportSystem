#!/usr/bin/env python3
"""Build the submission artifacts from a held-out sample of the TEST split.

Produces (default ``outputs/submission/``):

    queries.csv       query_id,query         — Q001… ids, fixed seed
    gold.csv          query_id,query,intent,category,response,flags
    predictions.csv   query_id,predicted_intent,predicted_category,decision
    responses.jsonl   generated responses (one JSON object per line)
    traces.jsonl      full five-agent traces (one JSON object per line)

The prediction CSV holds *only* the four required columns; responses and traces
live in their own files, as the submission format requires.

Usage::

    .venv/bin/python scripts/make_submission.py                  # 300 rows
    .venv/bin/python scripts/make_submission.py --n 500 --seed 13
    .venv/bin/python scripts/make_submission.py --baseline       # Day-1 path
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                     # noqa: E402

from nlp.cli import write_outputs                       # noqa: E402
from nlp.pipeline import Level2Pipeline                 # noqa: E402

OUT_DIR = "outputs/submission"


def build_queries(n: int, seed: int, split: str = "test") -> tuple[list[dict], pd.DataFrame]:
    """Deterministic sample of *n* rows: Q001… ids, sorted by TEST row order."""
    path = os.path.join("outputs", "splits", f"{split}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"error: {path} missing — run the split script first")

    df = pd.read_csv(path)
    if n > len(df):
        raise SystemExit(f"error: --n {n} exceeds the {len(df)} rows in {path}")

    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(df)), n))   # sorted => reproducible ids

    rows = []
    for i, idx in enumerate(picked, start=1):
        row = df.iloc[idx]
        rows.append({
            "query_id": f"Q{i:03d}",
            "query": str(row["instruction"]).strip(),
            "intent": str(row["intent"]),
            "category": str(row["category"]),
            "response": str(row["response"]),
            "flags": str(row["flags"]),
            "test_row": idx,
        })
    return rows, df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300, help="rows to sample (default 300)")
    ap.add_argument("--seed", type=int, default=13, help="sampling seed (default 13)")
    ap.add_argument("--split", default="test", help="split to sample (default test)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--baseline", action="store_true",
                    help="generate with the Day-1 A1 -> A2 path")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    queries, df = build_queries(args.n, args.seed, args.split)
    print(f"sampled {len(queries)} rows from {args.split} "
          f"(seed={args.seed}, {len(df)} available)")

    # ── held-out query file + gold file ──────────────────────────────
    q_path = os.path.join(out_dir, "queries.csv")
    with open(q_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "query"])
        for r in queries:
            w.writerow([r["query_id"], r["query"]])

    gold_path = os.path.join(out_dir, "gold.csv")
    with open(gold_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "query", "intent", "category", "response", "flags"])
        for r in queries:
            w.writerow([r["query_id"], r["query"], r["intent"],
                        r["category"], r["response"], r["flags"]])

    # ── run the pipeline ─────────────────────────────────────────────
    pipe = Level2Pipeline()
    results = []
    t_start = time.perf_counter()
    for i, r in enumerate(queries, start=1):
        res = pipe.resolve(r["query"], baseline=args.baseline, with_map=False)
        res["query_id"] = r["query_id"]
        res["gold_intent"] = r["intent"]
        res["gold_category"] = r["category"]
        results.append(res)
        if i % 25 == 0 or i == len(queries):
            print(f"  [{i}/{len(queries)}]", file=sys.stderr)

    elapsed = time.perf_counter() - t_start

    # ── artifacts ────────────────────────────────────────────────────
    write_outputs(
        results,
        os.path.join(out_dir, "predictions.csv"),
        os.path.join(out_dir, "responses.jsonl"),
        os.path.join(out_dir, "traces.jsonl"),
    )

    # ── report ───────────────────────────────────────────────────────
    intent_acc = sum(r["predicted_intent"] == r["gold_intent"]
                     for r in results) / len(results)
    cat_acc = sum(r["predicted_category"] == r["gold_category"]
                  for r in results) / len(results)
    decisions = Counter(r.get("decision") or "BASELINE" for r in results)
    answered = [r for r in results if r.get("decision") == "ANSWER"]
    answered_acc = (sum(r["predicted_intent"] == r["gold_intent"]
                        for r in answered) / len(answered)) if answered else 0.0

    q = sorted(r["total_latency_ms"] for r in results)
    p50 = q[len(q) // 2]
    p95 = q[max(0, int(len(q) * 0.95) - 1)]

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": args.split,
        "seed": args.seed,
        "n": len(results),
        "mode": "baseline" if args.baseline else "l2",
        "intent_accuracy": round(intent_acc, 4),
        "category_accuracy": round(cat_acc, 4),
        "intent_accuracy_on_answered": round(answered_acc, 4),
        "decisions": dict(decisions),
        "latency_ms": {"p50": p50, "p95": p95, "total_s": round(elapsed, 1)},
        "files": {
            "queries": q_path,
            "gold": gold_path,
            "predictions": os.path.join(out_dir, "predictions.csv"),
            "responses": os.path.join(out_dir, "responses.jsonl"),
            "traces": os.path.join(out_dir, "traces.jsonl"),
        },
    }
    with open(os.path.join(out_dir, "submission_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(f"intent accuracy   : {intent_acc:.4f}")
    print(f"category accuracy : {cat_acc:.4f}")
    print(f"answered accuracy : {answered_acc:.4f} (n={len(answered)})")
    print("decisions         : " + ", ".join(
        f"{k}={v}" for k, v in sorted(decisions.items())))
    print(f"latency           : p50={p50:.1f}ms p95={p95:.1f}ms")
    print(f"\nwrote {q_path}")
    print(f"wrote {gold_path}")
    print(f"wrote {out_dir}/predictions.csv, responses.jsonl, traces.jsonl")
    print(f"wrote {out_dir}/submission_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
