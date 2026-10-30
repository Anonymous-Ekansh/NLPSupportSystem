#!/usr/bin/env python3
"""Validate a submission predictions CSV against the required format.

Checks
------
* header is exactly ``query_id,predicted_intent,predicted_category,decision``
  (same names, same order, no extra/missing columns)
* ids are unique, non-empty and match the gold/query file when one is given
* every ``predicted_intent`` is an exact dataset intent label
* every ``predicted_category`` is an exact dataset category label
* every ``decision`` is one of ANSWER / CLARIFY / ESCALATE
* no row is blank, row counts agree

Usage::

    .venv/bin/python scripts/validate_submission.py \
        outputs/submission/predictions.csv \
        --gold outputs/submission/gold.csv

Exit code 0 = valid, 1 = invalid (every problem is printed).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_COLUMNS = ["query_id", "predicted_intent",
                    "predicted_category", "decision"]
DECISIONS = {"ANSWER", "CLARIFY", "ESCALATE"}
POLICY_PATH = "config/l2_policy.json"


def load_labels() -> tuple[set[str], set[str]]:
    """Exact dataset labels, taken from the tuned policy's intent map."""
    with open(POLICY_PATH, encoding="utf-8") as fh:
        policy = json.load(fh)
    intent2cat = policy.get("intents") or {}
    if not intent2cat:
        raise SystemExit(f"error: no 'intents' map in {POLICY_PATH}")
    return set(intent2cat), set(intent2cat.values())


def validate(path: str, gold_path: str | None = None) -> list[str]:
    """Return a list of problems (empty means valid)."""
    problems: list[str] = []

    if not os.path.exists(path):
        return [f"missing file: {path}"]

    intents, categories = load_labels()

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [f"{path} is empty"]
        rows = list(reader)

    if header != REQUIRED_COLUMNS:
        problems.append(
            f"header must be exactly {REQUIRED_COLUMNS}, got {header}")

    seen: set[str] = set()
    blank_rows = 0
    for lineno, row in enumerate(rows, start=2):
        if not row or all(not c.strip() for c in row):
            blank_rows += 1
            continue
        if len(row) != len(REQUIRED_COLUMNS):
            problems.append(f"line {lineno}: expected {len(REQUIRED_COLUMNS)} "
                            f"fields, got {len(row)}")
            continue

        query_id, intent, category, decision = [c.strip() for c in row]

        if not query_id:
            problems.append(f"line {lineno}: empty query_id")
        elif query_id in seen:
            problems.append(f"line {lineno}: duplicate query_id {query_id!r}")
        seen.add(query_id)

        if intent not in intents:
            problems.append(f"line {lineno}: predicted_intent {intent!r} is "
                            f"not a dataset label")
        if category not in categories:
            problems.append(f"line {lineno}: predicted_category {category!r} "
                            f"is not a dataset category")
        if decision not in DECISIONS:
            problems.append(f"line {lineno}: decision {decision!r} not in "
                            f"{sorted(DECISIONS)}")

    if blank_rows:
        problems.append(f"{blank_rows} blank row(s)")

    if not rows:
        problems.append("no data rows")

    # ── cross-check against the gold / query file ──────────────────
    if gold_path:
        if not os.path.exists(gold_path):
            problems.append(f"missing gold file: {gold_path}")
        else:
            with open(gold_path, newline="", encoding="utf-8") as fh:
                gold_rows = list(csv.DictReader(fh))
            gold_ids = [str(r.get("query_id", "")).strip() for r in gold_rows]
            if len(set(gold_ids)) != len(gold_ids):
                problems.append("gold file has duplicate query_ids")
            missing = sorted(set(gold_ids) - seen)
            extra = sorted(seen - set(gold_ids))
            if missing:
                problems.append(f"{len(missing)} gold ids missing from "
                                f"predictions (e.g. {missing[:5]})")
            if extra:
                problems.append(f"{len(extra)} prediction ids not in gold "
                                f"(e.g. {extra[:5]})")
            if len(gold_ids) != len(seen):
                problems.append(f"row count mismatch: predictions={len(seen)} "
                                f"gold={len(gold_ids)}")

    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", default="outputs/submission/predictions.csv")
    ap.add_argument("--gold", default=None,
                    help="gold/queries CSV to cross-check ids and counts")
    args = ap.parse_args(argv)

    intents, categories = load_labels()
    problems = validate(args.csv, args.gold)

    if problems:
        print(f"INVALID: {args.csv}")
        for p in problems[:40]:
            print(f"  - {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1

    print(f"VALID: {args.csv}")
    print(f"  rows          : {len(rows)}")
    print(f"  intent labels : {intents.__len__()} allowed, "
          f"{len({r['predicted_intent'] for r in rows})} used")
    print(f"  categories    : {categories.__len__()} allowed, "
          f"{len({r['predicted_category'] for r in rows})} used")
    print("  decisions     : " + ", ".join(f"{k}={v}"
                                            for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
