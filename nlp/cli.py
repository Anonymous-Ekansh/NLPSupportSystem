#!/usr/bin/env python3
"""Command-line interface for the Level-2 support pipeline.

Single query (pretty five-agent trace)::

    .venv/bin/python -m nlp.cli "I need a refund but cannot find my order number"
    .venv/bin/python -m nlp.cli --json "where is my order"
    .venv/bin/python -m nlp.cli --follow-up "123456" "I need a refund but cannot find my order number"
    .venv/bin/python -m nlp.cli --baseline "where is my order"     # Day-1 A1 -> A2

Batch over a CSV (any of the columns ``query`` / ``instruction`` / ``text``)::

    .venv/bin/python -m nlp.cli --batch queries.csv \
        --out-csv predictions.csv --out-responses responses.jsonl \
        --out-traces traces.jsonl

Exit codes: 0 success, 2 bad input, 1 internal error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.pipeline import Level2Pipeline, _format_trace  # noqa: E402

QUERY_COLUMNS = ("query", "instruction", "text", "message")
MAX_BATCH = 5000


# ── pretty printing ──────────────────────────────────────────────────────

def print_trace(result: dict, show_payloads: bool = False) -> None:
    """Human-readable decision + one line per agent."""
    print(_format_trace(result))

    a3 = result.get("a3") or {}
    if a3.get("reasons"):
        print("\nA3 reasons:")
        for reason in a3["reasons"]:
            print(f"  • {reason}")

    if a3.get("missing_slots"):
        print("\nMissing information:")
        for slot in a3["missing_slots"]:
            print(f"  • {slot.get('label')} ({slot.get('slot')}) — "
                  f"{slot.get('question')}")

    if a3.get("clarify_options"):
        print("\nClarify options:")
        for opt in a3["clarify_options"]:
            print(f"  • {opt.get('intent')} (p={opt.get('prob')})")

    a4 = result.get("a4")
    if a4:
        print(f"\nA4 verdict: {a4['verdict']}")
        for chk in a4.get("checks", []):
            mark = "ok" if chk.get("passed") else "FAIL"
            print(f"  [{mark}] {chk.get('name')}: {chk.get('detail')}")
        if a4.get("issues"):
            print("  issues: " + "; ".join(a4["issues"]))
        if a4.get("revised_response"):
            print("  revised: " + a4["revised_response"][:300])

    if show_payloads:
        print("\nAgent payloads:")
        for step in result.get("agent_trace") or []:
            print(f"\n── {step['agent']} ({step.get('role')}) "
                  f"decision={step.get('decision')} ──")
            print(json.dumps(step.get("payload", {}), indent=2, default=str)[:2000])


# ── batch ────────────────────────────────────────────────────────────────

def _detect_query_column(fieldnames: list[str]) -> str:
    for col in QUERY_COLUMNS:
        if col in fieldnames:
            return col
    raise SystemExit(
        f"error: input CSV needs one of the columns {QUERY_COLUMNS}; "
        f"found {fieldnames}")


PREDICTION_COLUMNS = ("query_id", "predicted_intent",
                      "predicted_category", "decision")


def write_outputs(results: list[dict], out_csv: str, out_responses: str,
                  out_traces: str) -> None:
    """Write the three submission artifacts from pipeline results.

    The predictions CSV holds exactly `query_id, predicted_intent,
    predicted_category, decision`; generated responses and agent traces go to
    their own JSONL files so nothing long-form lands in the CSV.
    """
    for path in (out_csv, out_responses, out_traces):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(PREDICTION_COLUMNS)
        for res in results:
            writer.writerow([res["query_id"], res.get("predicted_intent") or "",
                             res.get("predicted_category") or "",
                             res.get("decision") or "ANSWER"])

    with open(out_responses, "w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps({
                "query_id": res["query_id"],
                "query": res["query"],
                "decision": res.get("decision"),
                "response": res.get("final_response", ""),
                "predicted_intent": res.get("predicted_intent"),
                "predicted_category": res.get("predicted_category"),
                "reason_codes": res.get("reason_codes", []),
            }, ensure_ascii=False) + "\n")

    with open(out_traces, "w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps({
                "query_id": res["query_id"],
                "query": res["query"],
                "mode": res.get("mode"),
                "total_latency_ms": res.get("total_latency_ms"),
                "trace": res.get("trace", []),
            }, ensure_ascii=False, default=str) + "\n")


def run_batch(args) -> int:
    with open(args.batch, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit("error: empty CSV")
        qcol = _detect_query_column(list(reader.fieldnames))
        rows = list(reader)

    if not rows:
        raise SystemExit("error: no rows in input CSV")
    if len(rows) > MAX_BATCH:
        raise SystemExit(f"error: {len(rows)} rows exceeds the {MAX_BATCH} limit")

    pipe = Level2Pipeline()
    results = []
    for n, row in enumerate(rows, start=1):
        query = (row.get(qcol) or "").strip()
        query_id = (row.get("query_id") or f"Q{n:03d}").strip()
        if not query:
            continue
        res = pipe.resolve(query, baseline=args.baseline, with_map=False)
        res["query_id"] = query_id
        results.append(res)
        print(f"[{n}/{len(rows)}] {query_id} -> {res.get('decision') or 'BASELINE'}",
              file=sys.stderr)

    write_outputs(results, args.out_csv, args.out_responses, args.out_traces)

    counted: dict[str, int] = {}
    for res in results:
        key = res.get("decision") or "BASELINE"
        counted[key] = counted.get(key, 0) + 1
    print(f"\n{len(results)} queries -> {args.out_csv}")
    print("decisions: " + ", ".join(f"{k}={v}" for k, v in sorted(counted.items())))
    return 0


# ── argument parsing ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m nlp.cli",
        description="Run the Level-2 support pipeline (A1 -> A3 -> A2 -> A4 -> A5).")
    ap.add_argument("query", nargs="*", help="query text")
    ap.add_argument("--json", action="store_true", help="print the full result JSON")
    ap.add_argument("--baseline", action="store_true",
                    help="run the Day-1 A1 -> A2 path only")
    ap.add_argument("--payloads", action="store_true",
                    help="also dump each agent's raw payload")
    ap.add_argument("--follow-up", dest="follow_up", default=None,
                    metavar="ANSWER",
                    help="answer to a clarification question; re-runs the "
                         "pipeline on the query plus this answer")
    ap.add_argument("--batch", default=None, metavar="CSV",
                    help="run a CSV of queries")
    ap.add_argument("--out-csv", dest="out_csv", default="outputs/submission/predictions.csv")
    ap.add_argument("--out-responses", dest="out_responses",
                    default="outputs/submission/responses.jsonl")
    ap.add_argument("--out-traces", dest="out_traces",
                    default="outputs/submission/traces.jsonl")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.batch:
        return run_batch(args)

    query = " ".join(args.query).strip()
    if not query:
        build_parser().print_help()
        return 2

    pipe = Level2Pipeline()
    if args.follow_up:
        result = pipe.resolve_with_followup(query, args.follow_up,
                                            baseline=args.baseline)
    else:
        result = pipe.resolve(query, baseline=args.baseline)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_trace(result, show_payloads=args.payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
