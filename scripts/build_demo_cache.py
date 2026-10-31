#!/usr/bin/env python3
"""Regenerate `web/public/demo_cache.json` from the real Level-2 pipeline.

The cache is the UI's offline fallback: when the API is cold or unreachable the
web app serves one of these exact-key runs and labels it **Cached**. Every
entry here is a genuine pipeline output, never a hand-written example.

Usage::

    .venv/bin/python scripts/build_demo_cache.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.pipeline import Level2Pipeline  # noqa: E402

CACHE_PATH = "web/public/demo_cache.json"

# The demo scenarios the UI offers, plus the original Day-1 cache keys so no
# existing entry is lost.
SCENARIOS = [
    "How can I request a refund for my purchase?",              # normal refund
    "I need a refund but cannot find my order number",          # missing detail
    "refund",                                                   # ambiguous
    "i wanna cancel my order pls its been a week",              # typo / slang
    "bhai mera refund kab aayega please jaldi batao",           # Hinglish
    "asdkjhqwoieu 🎉🎉",                                          # gibberish
    "I will sue you over this charge",                          # legal threat
    # previously cached Day-1 queries, regenerated with the full pipeline
    "How can I get a refund for my purchase?",
    "filing consumer claim against ur organizzation",
    "i need to check status for order {{Order Number}}",
    "it did not work",
    "what is the weather today",
    "how do I cancel my subscription?",
    "I want to cancel my order",
    "Where is my parcel? How long will delivery take?",
    "I forgot my password and can't log in",
]


def main() -> int:
    existing: dict = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as fh:
            existing = json.load(fh)

    keys = list(SCENARIOS)
    for key in existing:
        if key not in keys:
            keys.append(key)

    pipe = Level2Pipeline()
    cache: dict = {}
    for i, query in enumerate(keys, start=1):
        result = pipe.resolve(query)
        cache[query] = {
            "query": result["query"],
            "trace": result["trace"],
            "final_response": result["final_response"],
            "total_latency_ms": result["total_latency_ms"],
            "map": result["map"],
            "retrieval_backend": result["retrieval_backend"],
            "decision": result["decision"],
            "predicted_intent": result["predicted_intent"],
            "predicted_category": result["predicted_category"],
            "reason_codes": result["reason_codes"],
            "evidence_ids": result["evidence_ids"],
            "mode": result["mode"],
            "a1": result["a1"],
            "a3": result["a3"],
            "a4": result["a4"],
            "a5": result["a5"],
            "latency_breakdown_ms": result["latency_breakdown_ms"],
        }
        print(f"[{i}/{len(keys)}] {result['decision']:9s} {query[:60]}",
              file=sys.stderr)

    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)

    counts: dict[str, int] = {}
    for value in cache.values():
        key = value.get("decision") or "NONE"
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {CACHE_PATH}: {len(cache)} entries, "
          f"decisions {counts}, at {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
