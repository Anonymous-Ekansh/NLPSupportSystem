#!/usr/bin/env python
"""Run the finished Level-2 system over ONE file: the hand-written evaluation set.

Input   data/eval/evaluation_queries.csv (query_id, instruction; 4031 rows,
        NO gold labels — nothing here is tuned on or scored for accuracy).
Output  outputs/evaluation/
          predictions.csv   exact header query_id,predicted_intent,predicted_category,decision
          responses.jsonl   one record per query_id (decision, response, labels, reasons)
          responses.csv     same records as CSV
          traces.jsonl      full five-agent trace per query_id
          checkpoint.jsonl  per-row checkpoint (resumable)
          run_report.md / run_report.json

Contract honoured per row:
  * instruction is passed to the pipeline exactly as written ({{placeholders}},
    typos, slang — nothing stripped, rewritten or skipped);
  * every input row yields exactly one record in every output file;
  * predicted_intent / predicted_category are never blank (A1's labels via the
    pipeline; on an unexpected pipeline error the row is *not* skipped — A1 is
    re-run for its labels and the decision is forced to ESCALATE with the error
    recorded in that row's trace);
  * decisions are only ANSWER / CLARIFY / ESCALATE.

Modes:
  local (default)  Level2Pipeline.resolve(query, with_map=False), in-memory
                   retrieval, fully offline and deterministic.
  --api-url        POST /resolve per row against a running API, with retries,
                   waiting for /health to report ready first.

Usage:
  python scripts/run_evaluation.py \
      --input data/eval/evaluation_queries.csv --outdir outputs/evaluation \
      [--api-url URL --api-key KEY] [--limit N] [--resume]

Never tune on this file: no thresholds, configs or models are touched here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from statistics import quantiles

# Work from the repo root so relative paths (config/, outputs/, data/) resolve.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

CHECKPOINT_EVERY = 1          # append one row each time, so --resume never loses work
PROGRESS_EVERY = 50
# Same wording A5 uses for its own escalations, so fallback rows are consistent.
FALLBACK_ESCALATION = (
    "I'm passing this to a human specialist for review — I don't want to "
    "guess and get it wrong. No account, order, payment or refund changes "
    "have been made."
)


# ── input ────────────────────────────────────────────────────────────────

def read_input(path: str) -> list[dict]:
    """Read query_id + instruction (column may be `instruction` or `query`)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"ERROR: no rows in {path}")
    cols = [c.strip().lower() for c in (reader.fieldnames or [])]
    text_col = "instruction" if "instruction" in cols else \
        "query" if "query" in cols else None
    if text_col is None:
        raise SystemExit("ERROR: input needs an 'instruction' (or 'query') column")
    out = []
    for i, row in enumerate(rows, 1):
        out.append({
            "query_id": str(row.get("query_id", f"row{i:05d}")).strip(),
            "instruction": "" if row.get(text_col) is None else str(row[text_col]),
        })
    return out


# ── runners ──────────────────────────────────────────────────────────────

class LocalRunner:
    """Offline, deterministic: Level2Pipeline.resolve(with_map=False)."""

    def __init__(self):
        from nlp.pipeline import Level2Pipeline
        print("[run] building Level2Pipeline (in-memory retrieval) …")
        t0 = time.perf_counter()
        self.pipe = Level2Pipeline()
        _ = self.pipe.a1, self.pipe.a2, self.pipe.planner   # warm up once
        print(f"[run] pipeline ready in {time.perf_counter() - t0:.1f}s")

    def run_row(self, instruction: str) -> dict:
        result = self.pipe.resolve(instruction, with_map=False)
        return {
            "decision": result["decision"],
            "response": result["response"],
            "predicted_intent": result["predicted_intent"],
            "predicted_category": result["predicted_category"],
            "confidence": result["confidence"],
            "reason_codes": list(result["reason_codes"]),
            "trace": result["trace"],
            "latency_ms": result["total_latency_ms"],
            "error": None,
        }


class ApiRunner:
    """POST /resolve per row; waits for /health before starting."""

    def __init__(self, base_url: str, api_key: str | None):
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url.rstrip("/"),
                                    headers=headers, timeout=30.0)
        self._wait_for_health()
        # Warm the API path: the first /resolve call also pays model load.
        self._post("/resolve", {"query": "hello", "with_map": False}, retries=5)

    def _wait_for_health(self, timeout_s: float = 120.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                r = self._client.get("/health")
                data = r.json()
                if data.get("model_loaded"):
                    print(f"[run] API ready: {data.get('retrieval_backend')}")
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise SystemExit("ERROR: API did not report ready in time")

    def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                r = self._client.post(path, json=payload)
                if r.status_code == 200:
                    return r.json()
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except Exception as exc:            # transport-level retry
                last = exc
            time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"POST {path} failed after {retries} attempts: {last}")

    def run_row(self, instruction: str) -> dict:
        result = self._post("/resolve", {"query": instruction, "with_map": False})
        trace = result.get("trace") or []
        if not trace:                       # batch-style results may skip trace
            trace = (result.get("agent_trace") or [])
        return {
            "decision": result["decision"],
            "response": result.get("response") or result.get("final_response") or "",
            "predicted_intent": result["predicted_intent"],
            "predicted_category": result["predicted_category"],
            "confidence": result.get("confidence"),
            "reason_codes": list(result.get("reason_codes") or []),
            "trace": trace,
            "latency_ms": result.get("total_latency_ms"),
            "error": None,
        }


def fallback_record(runner, instruction: str, exc: Exception) -> dict:
    """A raised row is NOT skipped: reuse A1's labels, force ESCALATE."""
    from nlp.pipeline import Level2Pipeline
    pipe = runner.pipe if isinstance(runner, LocalRunner) else Level2Pipeline()
    a1_output = None
    try:
        a1_output = pipe.a1.analyze(instruction)
        intent, category = a1_output.get("intent", ""), a1_output.get("category", "")
        confidence = a1_output.get("confidence")
    except Exception as a1_exc:             # even A1 failed — never blank, but flag it
        intent, category, confidence = "", "", None
        a1_output = {"error": f"A1 unavailable: {a1_exc}"}
    error_step = {"agent": "RUNNER", "role": "error fallback",
                  "input": {"query": instruction},
                  "output": {"error": f"{type(exc).__name__}: {exc}"},
                  "decision": "error_fallback", "evidence": [], "latency_ms": 0}
    trace = [{
        "agent": "A1", "role": "Intent analyst",
        "input": {"query": instruction}, "output": a1_output,
        "decision": "error_fallback", "evidence": [], "latency_ms": 0,
    }] + [{
        "agent": agent, "role": role, "input": {}, "output": {
            "skipped": "pipeline error; decision forced to ESCALATE",
        },
        "decision": "error_fallback", "evidence": [], "latency_ms": 0,
    } for agent, role in (("A3", "Route planner"), ("A2", "Response specialist"),
                          ("A4", "Response verifier"), ("A5", "Coordinator"))]
    trace.append(error_step)
    return {
        "decision": "ESCALATE",
        "response": FALLBACK_ESCALATION,
        "predicted_intent": intent,
        "predicted_category": category,
        "confidence": confidence,
        "reason_codes": ["error_fallback"],
        "trace": trace,
        "latency_ms": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


# ── reports ──────────────────────────────────────────────────────────────

def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) < 20:
        return vals[-1]
    return float(quantiles(vals, n=20)[18])   # 95th percentile


def write_report(outdir: str, rows: list[dict], records: list[dict],
                 runtime_s: float) -> None:
    latencies = [r["latency_ms"] for r in records if isinstance(r["latency_ms"], (int, float))]
    decisions: dict[str, int] = {}
    reasons: dict[str, int] = {}
    missing_slot_clarifies = 0
    for rec in records:
        decisions[rec["decision"]] = decisions.get(rec["decision"], 0) + 1
        for code in rec["reason_codes"]:
            reasons[code] = reasons.get(code, 0) + 1
        if rec["decision"] == "CLARIFY" and "missing_information" in rec["reason_codes"]:
            missing_slot_clarifies += 1
    intent_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for rec in records:
        intent_counts[rec["predicted_intent"]] = intent_counts.get(rec["predicted_intent"], 0) + 1
        category_counts[rec["predicted_category"]] = category_counts.get(rec["predicted_category"], 0) + 1
    top10 = sorted(intent_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    low_conf = sum(1 for r in records
                   if isinstance(r["confidence"], (int, float)) and r["confidence"] < 0.6)
    placeholder_rows = [
        (row, rec) for row, rec in zip(rows, records) if "{{" in row["instruction"]
    ]
    placeholder_decisions: dict[str, int] = {}
    for _row, rec in placeholder_rows:
        placeholder_decisions[rec["decision"]] = placeholder_decisions.get(rec["decision"], 0) + 1
    error_rows = [rec for rec in records if rec["error"]]
    escalate_rows = [(row, rec) for row, rec in zip(rows, records)
                     if rec["decision"] == "ESCALATE"]

    summary = {
        "input_file": rows[0].get("input_file", "data/eval/evaluation_queries.csv"),
        "total_rows": len(rows),
        "error_rows": len(error_rows),
        "runtime_seconds": round(runtime_s, 1),
        "latency_ms": {
            "p50": round(_p50(latencies), 1),
            "p95": round(_p95(latencies), 1),
            "max": round(max(latencies), 1) if latencies else 0,
        },
        "decisions": decisions,
        "decision_shares": {d: round(c / len(rows), 4) for d, c in decisions.items()},
        "top10_intents": top10,
        "category_distribution": dict(sorted(category_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "reason_code_counts": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "clarify_from_missing_slots": missing_slot_clarifies,
        "placeholder_rows": len(placeholder_rows),
        "placeholder_decisions": placeholder_decisions,
        "confidence_below_0_6": low_conf,
        "escalate_rows": [
            {"query_id": row["query_id"], "instruction": row["instruction"],
             "reason_codes": rec["reason_codes"]}
            for row, rec in escalate_rows
        ],
        "note": "This file has NO gold labels: no accuracy or F1 is reported here.",
    }

    with open(os.path.join(outdir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    lines = [
        "# Evaluation run report — `evaluation_queries.csv` (no gold labels)",
        "",
        "**No accuracy or F1 is reported for this file: it has no gold labels.**",
        "",
        "| metric | value |",
        "|---|---|",
        f"| total rows | {len(rows)} |",
        f"| error/fallback rows | {len(error_rows)} (expected 0) |",
        f"| runtime | {runtime_s:.1f}s |",
        f"| latency p50 / p95 / max | {summary['latency_ms']['p50']} / "
        f"{summary['latency_ms']['p95']} / {summary['latency_ms']['max']} ms |",
        f"| low-confidence rows (< 0.6) | {low_conf} |",
        f"| rows containing \\u007b\\u007bplaceholders\\u007d\\u007d | "
        f"{len(placeholder_rows)} ({len(placeholder_rows) / len(rows):.1%}) " +
        (f"→ {placeholder_decisions}" if placeholder_decisions else "") + " |",
        "| clarify sourced from missing slots | missing_information: "
        f"{missing_slot_clarifies} |",
        "",
        "## Decisions",
        "",
        "| decision | count | share |",
        "|---|---|---|",
    ]
    for d in ("ANSWER", "CLARIFY", "ESCALATE"):
        c = decisions.get(d, 0)
        lines.append(f"| {d} | {c} | {c / len(rows):.4f} |")
    lines += ["", "## Top 10 predicted intents", "",
              "| intent | count |", "|---|---|"]
    for intent, n in top10:
        lines.append(f"| {intent} | {n} |")
    lines += ["", "## Category distribution", "",
              "| category | count |", "|---|---|"]
    for cat, n in summary["category_distribution"].items():
        lines.append(f"| {cat} | {n} |")
    lines += ["", "## Reason codes", "", "| reason code | count |", "|---|---|"]
    for code, n in summary["reason_code_counts"].items():
        lines.append(f"| {code} | {n} |")
    lines += [
        "",
        "## Known limitation of this run",
        "",
        "The hand-written evaluation file contains no gold labels, so intent "
        "accuracy is not measurable here — this run only proves the finished "
        "system produces complete, well-formed, identical decisions offline "
        "and via the API (see validate_evaluation.py).",
        "",
        "## Escalated rows", "",
    ]
    for row, rec in escalate_rows:
        codes = ", ".join(rec["reason_codes"])
        lines.append(f"- **{row['query_id']}** `{row['instruction']}` — {codes}")
    with open(os.path.join(outdir, "run_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    vit = sorted(values)
    mid = len(vit) // 2
    return float(vit[mid]) if len(vit) % 2 else float((vit[mid - 1] + vit[mid]) / 2)


# ── writers ──────────────────────────────────────────────────────────────

def write_outputs(outdir: str, rows: list[dict], records: list[dict]) -> None:
    # predictions.csv — exact header, UTF-8 without BOM, input order
    with open(os.path.join(outdir, "predictions.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "predicted_intent", "predicted_category", "decision"])
        for row, rec in zip(rows, records):
            w.writerow([row["query_id"], rec["predicted_intent"],
                        rec["predicted_category"], rec["decision"]])

    # responses.jsonl / responses.csv
    with open(os.path.join(outdir, "responses.jsonl"), "w", encoding="utf-8") as jf, \
         open(os.path.join(outdir, "responses.csv"), "w", newline="",
              encoding="utf-8") as cf:
        kw = csv.DictWriter(cf, fieldnames=[
            "query_id", "instruction", "decision", "response",
            "predicted_intent", "predicted_category", "confidence", "reason_codes"])
        kw.writeheader()
        for row, rec in zip(rows, records):
            slim = {
                "query_id": row["query_id"],
                "instruction": row["instruction"],
                "decision": rec["decision"],
                "response": rec["response"],
                "predicted_intent": rec["predicted_intent"],
                "predicted_category": rec["predicted_category"],
                "confidence": rec["confidence"],
                "reason_codes": rec["reason_codes"],
            }
            jf.write(json.dumps(slim, ensure_ascii=False) + "\n")
            kw.writerow({**slim, "reason_codes": ";".join(rec["reason_codes"])})

    # traces.jsonl — one full five-agent trace per query_id
    with open(os.path.join(outdir, "traces.jsonl"), "w", encoding="utf-8") as f:
        for row, rec in zip(rows, records):
            f.write(json.dumps({
                "query_id": row["query_id"],
                "instruction": row["instruction"],
                "trace": rec["trace"],
            }, ensure_ascii=False) + "\n")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="data/eval/evaluation_queries.csv")
    ap.add_argument("--outdir", default="outputs/evaluation")
    ap.add_argument("--api-url", default=None)
    ap.add_argument("--api-key", default=os.environ.get("EVALUATION_API_KEY"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t_start = time.perf_counter()
    rows = read_input(args.input)
    for row in rows:
        row["input_file"] = args.input
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(args.outdir, exist_ok=True)

    ckpt_path = os.path.join(args.outdir, "checkpoint.jsonl")
    done: dict[str, dict] = {}
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done[entry["query_id"]] = entry["record"]
        print(f"[run] resume: {len(done)} rows already in checkpoint")

    pending = [row for row in rows if row["query_id"] not in done]
    runner = (ApiRunner(args.api_url, args.api_key) if args.api_url
              else LocalRunner())

    errors = 0
    ckpt_file = open(ckpt_path, "a", encoding="utf-8")
    t0 = time.perf_counter()
    for i, row in enumerate(pending, 1):
        try:
            record = runner.run_row(row["instruction"])
        except Exception as exc:            # never skip: ESCALATE fallback
            record = fallback_record(runner, row["instruction"], exc)
            errors += 1
            print(f"[run] fallback row {row['query_id']}: {record['error']}")
        record["query_id"] = row["query_id"]
        record["instruction"] = row["instruction"]
        done[row["query_id"]] = record
        ckpt_file.write(json.dumps({"query_id": row["query_id"],
                                    "record": record}, ensure_ascii=False) + "\n")
        if i % CHECKPOINT_EVERY == 0:
            ckpt_file.flush()
        if i % PROGRESS_EVERY == 0 or i == len(pending):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            eta = (len(pending) - i) / max(rate, 1e-9)
            print(f"[run] {i}/{len(pending)} rows  ({rate:.1f} rows/s, "
                  f"ETA {eta:.0f}s, errors so far {errors})")
    ckpt_file.flush()
    ckpt_file.close()

    records = [done[row["query_id"]] for row in rows]
    runtime_s = time.perf_counter() - t_start
    write_outputs(args.outdir, rows, records)
    write_report(args.outdir, rows, records, runtime_s)
    print(f"[run] wrote outputs to {args.outdir} "
          f"(errors: {errors}, total {runtime_s:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
