#!/usr/bin/env python
"""Validate the evaluation-run submission artifacts.  Tests T1–T10 only.

Usage:
    python scripts/validate_evaluation.py \
        --input data/eval/evaluation_queries.csv --outdir outputs/evaluation

Prints every problem found, then a T1–T10 pass/fail table, then exits 1 if
anything failed, 0 otherwise.

Check-definition notes (kept visible on purpose):
  * T6 v1 required a CLARIFY response to END with "?"; that flagged 275 rows
    following the policy's own slot-clarify template ("Could you share your
    order number? You can find it ...") — a direct question plus one guidance
    sentence ending in ".".  0 of 421 CLARIFYs lack a question mark entirely,
    so the check is now "contains at least one question mark".  The v1 count
    stays in run_report.md.
  * T10 v1 compared API results against predictions.csv with the wrong key
    names (predicted_intent vs intent); every row actually matched. Fixed.

T9 and T10 notes:
  * T9 is explicitly a WIRING SANITY CHECK on rows the model has already seen
    (verbatim split rows) — it measures determinism/wiring, not accuracy.
  * T10 starts the local API (in-memory retrieval, Supabase env forced empty)
    on 127.0.0.1, sends 25 rows via POST /resolve and 25 via POST
    /resolve/batch, and compares intent/category/decision/response with the
    offline run for the same rows.  It is not an accuracy metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

VALID_DECISIONS = {"ANSWER", "CLARIFY", "ESCALATE"}
REQUIRED_AGENTS = ["A1", "A3", "A2", "A4", "A5"]
EXPECTED_HEADER = "query_id,predicted_intent,predicted_category,decision"

problems: dict[str, list[str]] = {}


def fail(check: str, message: str) -> None:
    problems.setdefault(check, []).append(message)
    print(f"  FAIL: {message}")


def head(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def _read_input(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"ERROR: input {path} has no rows")
    return rows


def _read_jsonl(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                fail("T5", f"{path} line {line_no}: invalid JSON ({exc})")
                continue
            out[rec.get("query_id", f"<line {line_no}>")] = rec
    return out


def _read_predictions(path: str) -> tuple[str, list[list[str]]]:
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        rows = [r for r in csv.reader(f) if r]
    return header, rows


def _pct(num: int, den: int) -> str:
    return f"{num / den:.4f}" if den else "0/0"


# ── T1 input integrity ───────────────────────────────────────────────────

def run_t1(input_rows: list[dict]) -> dict[str, str]:
    head("T1 Input integrity")
    texts_col = "instruction" if "instruction" in (input_rows[0] or {}) else "query"
    ids = [str(r.get("query_id", "")).strip() for r in input_rows]
    if len(input_rows) != 4031:
        fail("T1", f"expected 4031 rows, found {len(input_rows)}")
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        fail("T1", f"duplicate query_ids: {dupes[:10]}")
    empty_ids = [i + 1 for i, qid in enumerate(ids) if not qid]
    if empty_ids:
        fail("T1", f"empty query_ids at rows {empty_ids[:10]}")
    empty_texts = [i + 1 for i, r in enumerate(input_rows)
                   if not str(r.get(texts_col, "")).strip()]
    if empty_texts:
        fail("T1", f"empty instructions at rows {empty_texts[:10]}")
    print(f"  {len(input_rows)} rows, {len(set(ids))} unique ids, "
          f"text column {texts_col!r}")
    return {qid: str(r.get(texts_col, "")) for qid, r in zip(ids, input_rows)}


# ── T2 predictions.csv format ────────────────────────────────────────────

def run_t2(header: str, rows: list[list[str]],
           input_texts: dict[str, str]) -> dict[str, dict]:
    head("T2 predictions.csv format")
    if header != EXPECTED_HEADER:
        fail("T2", f"header is {header!r}, expected {EXPECTED_HEADER!r} "
                   "(exact, no extra columns, no index column)")
    if len(rows) != 4031:
        fail("T2", f"expected 4031 rows, found {len(rows)}")
    ids = [r[0] if r else "" for r in rows]
    if ids != list(input_texts.keys()):
        if set(ids) == set(input_texts):
            first_diff = next(i for i, (a, b) in enumerate(zip(ids, input_texts))
                              if a != b)
            fail("T2", f"query_ids match as a set but ORDER differs from the input "
                       f"(first difference at row {first_diff + 1})")
        else:
            missing = sorted(set(input_texts) - set(ids))
            extra = sorted(set(ids) - set(input_texts))
            fail("T2", f"ids differ from input; missing={missing[:10]} extra={extra[:10]}")
    blanks = [(i + 1, j + 1) for i, row in enumerate(rows) for j, cell in enumerate(row)
              if cell.strip() in ("", "nan", "NaN", "None", "null")]
    for row_no, col_no in blanks[:20]:
        fail("T2", f"row {row_no} col {col_no}: blank/NaN cell")
    widths = {len(row) for row in rows}
    if widths != {4}:
        fail("T2", f"every row must have exactly 4 columns, found widths {sorted(widths)}")
    bom = open(os.path.join(OUTDIR, "predictions.csv"), "rb").read(3) == b"\xef\xbb\xbf"
    if bom:
        fail("T2", "predictions.csv starts with a UTF-8 BOM (must be BOM-less)")
    print(f"  {len(rows)} rows, header {'matches' if header == EXPECTED_HEADER else 'BAD'}")
    return {r[0]: {"intent": r[1], "category": r[2], "decision": r[3]}
            for r in rows if len(r) == 4}


# ── T3 dataset labels & mapping ──────────────────────────────────────────

def run_t3(predictions: dict[str, dict]) -> None:
    head("T3 Dataset labels & category mapping")
    with open("config/l2_policy.json", encoding="utf-8") as f:
        intent_to_cat = json.load(f)["intents"]
    if len(intent_to_cat) != 27:
        fail("T3", f"policy intents map has {len(intent_to_cat)} intents, expected 27")
    bad_intent = sorted({rec["intent"] for rec in predictions.values()}
                        - set(intent_to_cat))
    if bad_intent:
        fail("T3", f"predicted_intents that are not dataset intents: {bad_intent[:10]}")
    bad_cat = sorted({rec["category"] for rec in predictions.values()}
                     - set(intent_to_cat.values()))
    if bad_cat:
        fail("T3", f"predicted_categories that are not dataset categories: {bad_cat[:10]}")
    mismatched = [(qid, rec) for qid, rec in predictions.items()
                  if intent_to_cat.get(rec["intent"]) != rec["category"]]
    for qid, rec in mismatched[:20]:
        fail("T3", f"{qid}: intent {rec['intent']} maps to "
                   f"{intent_to_cat.get(rec['intent'])!r} but category is {rec['category']!r}")
    print(f"  {len(predictions)} label tuples checked "
          f"({len(bad_intent)} bad intents, {len(bad_cat)} bad categories, "
          f"{len(mismatched)} mapping mismatches)")


# ── T4 decision domain ───────────────────────────────────────────────────

def run_t4(predictions: dict[str, dict], responses: dict[str, dict]) -> None:
    head("T4 Decisions")
    bad = sorted({rec["decision"] for rec in predictions.values()} - VALID_DECISIONS)
    if bad:
        fail("T4", f"invalid decisions in predictions.csv: {bad[:10]}")
    resp_bad = sorted({rec["decision"] for rec in responses.values()
                       if rec.get("decision")} - VALID_DECISIONS)
    if resp_bad:
        fail("T4", f"invalid decisions in responses.jsonl: {resp_bad[:10]}")
    print("  all decisions in {ANSWER, CLARIFY, ESCALATE}" if not (bad or resp_bad)
          else "  see failures above")


# ── T5 completeness of responses & traces ────────────────────────────────

def run_t5(input_texts: dict[str, str], responses: dict[str, dict],
           traces: dict[str, dict]) -> None:
    head("T5 Completeness (responses.jsonl + traces.jsonl)")
    ids = list(input_texts.keys())
    for name, store in (("responses.jsonl", responses), ("traces.jsonl", traces)):
        if len(store) != len(ids):
            fail("T5", f"{name}: {len(store)} records, expected exactly {len(ids)}")
        missing = [qid for qid in ids if qid not in store]
        if missing:
            fail("T5", f"{name} missing query_ids: {missing[:10]}")
    for qid in ids:
        trace = (traces.get(qid, {}).get("trace") or [])
        agents = [str(step.get("agent")) for step in trace]
        idx = [agents.index(a) if a in agents else -1 for a in REQUIRED_AGENTS]
        if any(i == -1 for i in idx):
            missing = [a for a, i in zip(REQUIRED_AGENTS, idx) if i == -1]
            fail("T5", f"{qid}: trace missing agents {missing}")
        elif idx != sorted(idx):
            fail("T5", f"{qid}: agents out of required order (got {agents})")
        for step in trace:
            for field in ("role", "input", "output", "decision", "evidence"):
                if field not in step:
                    fail("T5", f"{qid} step {step.get('agent')}: missing field {field!r}")
                    break
        resp = responses.get(qid)
        if resp is None:
            fail("T5", f"responses.jsonl missing {qid}")
        elif not str(resp.get("response", "")).strip():
            fail("T5", f"{qid}: response is empty")
    print(f"  checked {len(ids)} jsonl records and traces")


# ── T6 responsible behaviour ─────────────────────────────────────────────

def run_t6(responses: dict[str, dict]) -> None:
    head("T6 Responsible behaviour")
    from nlp.l2.a4 import claim_matches

    claim_hits: list[str] = []
    brace_hits: list[str] = []
    answer_blank: list[str] = []
    clarify_not_question: list[tuple[str, str]] = []
    escalate_bad_text: list[tuple[str, str]] = []
    for qid, rec in responses.items():
        resp = str(rec.get("response", ""))
        decision = rec.get("decision")
        if claim_matches(resp):
            claim_hits.append(f"{qid}: {str(rec.get('predicted_intent'))}")
        if "{{" in resp:
            brace_hits.append(qid)
        if decision == "ANSWER" and not resp.strip():
            answer_blank.append(qid)
        if decision == "CLARIFY" and "?" not in resp:
            clarify_not_question.append((qid, resp))
        if decision == "ESCALATE":
            low = resp.lower()
            human_review = ("human" in low and
                            ("review" in low or "specialist" in low))
            nothing_changed = ("no account, order, payment or refund changes"
                               in low and "have been made" in low) or \
                              ("haven't made any" in low) or \
                              ("nothing has been changed" in low)
            if not (human_review and nothing_changed):
                escalate_bad_text.append((qid, resp))
    for qid, intent in claim_hits:
        fail("T6", f"A4 completion-claim detector matched in response of {qid} ({intent})")
    for qid in brace_hits:
        fail("T6", f"unresolved \"{{{{\" left in response of {qid}")
    for qid in answer_blank:
        fail("T6", f"ANSWER with empty response: {qid}")
    for qid, resp in clarify_not_question:
        fail("T6", f"CLARIFY response is not a question ({qid}): {resp!r}")
    for qid, resp in escalate_bad_text:
        fail("T6", f"ESCALATE message lacks human-review/nothing-changed wording "
                   f"({qid}): {resp!r}")
    print(f"  {len(responses)} responses scanned with the A4 detector: "
          f"claims={len(claim_hits)} braces={len(brace_hits)} "
          f"blank-answers={len(answer_blank)} "
          f"clarify-not-question={len(clarify_not_question)} "
          f"weak-escalate={len(escalate_bad_text)}")


# ── T7 determinism (re-run 300 seeded rows) ──────────────────────────────

def run_t7(input_texts: dict[str, str],
           predictions: dict[str, dict]) -> None:
    head("T7 Determinism (re-run 300 seeded random rows)")
    rng = random.Random(2026)
    sample = rng.sample(list(input_texts.items()), 300)
    from nlp.pipeline import Level2Pipeline
    print("  building a fresh pipeline …")
    pipe = Level2Pipeline()
    _ = pipe.a1, pipe.a2, pipe.planner
    mism = 0
    t0 = time.perf_counter()
    for k, (qid, text) in enumerate(sample, 1):
        out = pipe.resolve(text, with_map=False)
        got = (out["predicted_intent"], out["predicted_category"], out["decision"])
        pred = predictions[qid]
        want = (pred["intent"], pred["category"], pred["decision"])
        if got != want:
            mism += 1
            fail("T7", f"{qid}: re-run {got} != run-1 {want}")
        if k % 100 == 0:
            print(f"  {k}/300 re-run ({time.perf_counter() - t0:.0f}s)")
    if mism == 0:
        print(f"  300/300 identical ({time.perf_counter() - t0:.0f}s)")


# ── T8 duplicate consistency ─────────────────────────────────────────────

def run_t8(input_texts: dict[str, str], predictions: dict[str, dict],
           responses: dict[str, dict]) -> None:
    head("T8 Duplicate consistency")
    groups: dict[str, list[str]] = {}
    for qid, text in input_texts.items():
        groups.setdefault(text, []).append(qid)
    dupes = {t: ids for t, ids in groups.items() if len(ids) > 1}
    bad = 0
    for text, ids in dupes.items():
        tuples = {
            (predictions[qid]["intent"], predictions[qid]["category"],
             predictions[qid]["decision"],
             str(responses.get(qid, {}).get("response", "")))
            for qid in ids if qid in predictions
        }
        if len(tuples) > 1:
            bad += 1
            fail("T8", f"inconsistent outputs across duplicated rows {ids[:5]}")
    print(f"  {len(dupes)} duplicated instruction text(s), "
          f"{sum(len(v) for v in dupes.values())} rows; "
          f"{'all consistent' if bad == 0 else f'{bad} inconsistent group(s)'}")


# ── T9 wiring sanity (NOT accuracy) ──────────────────────────────────────

def run_t9(input_texts: dict[str, str], predictions: dict[str, dict]) -> None:
    head("T9 Wiring sanity — verbatim split rows (NOT accuracy)")
    label_counts: dict[str, dict[str, int]] = {}
    found_any = False
    for split in ("train", "dev", "test"):
        path = os.path.join("outputs", "splits", f"{split}.csv")
        if not os.path.exists(path):
            print(f"  (info) {path} not found, skipping")
            continue
        found_any = True
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                text = " ".join(
                    str(row.get("instruction", row.get("query", ""))).split()
                ).lower()
                intent = str(row.get("intent", "")).strip()
                if text:
                    label_counts.setdefault(text, {})
                    label_counts[text][intent] = label_counts[text].get(intent, 0) + 1
    if not found_any:
        print("  (info) outputs/splits/*.csv not found — T9 informational only")
        return
    single = {t: list(labels)[0] for t, labels in label_counts.items() if len(labels) == 1}
    matched = []
    for qid, text in input_texts.items():
        key = " ".join(text.split()).lower()
        if key in single and qid in predictions:
            matched.append((qid, single[key], predictions[qid]["intent"]))
    agree = sum(1 for _qid, label, got in matched if got == label)
    pct = _pct(agree, len(matched))
    print(f"  verbatim matches in splits with a single intent label: {len(matched)}; "
          f"predicted_intent agreement {agree}/{len(matched)} = {pct} "
          f"(wiring sanity only, not accuracy)")
    if matched and agree / len(matched) < 0.85:
        fail("T9", f"agreement {agree}/{len(matched)} possibly indicates broken wiring "
                   "or nondeterminism (this is a sanity check, not a quality metric)")


# ── T10 API smoke test ───────────────────────────────────────────────────

def run_t10(input_texts: dict[str, str], predictions: dict[str, dict],
            responses: dict[str, dict]) -> None:
    head("T10 API smoke test (25 single + 25 batch == offline)")
    import httpx

    chosen = None
    for port in range(8931, 8941):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                chosen = port
                break
            except OSError:
                continue
    if chosen is None:
        fail("T10", "no free local port 8931–8940")
        return
    # Empty-string Supabase env vars force the API into in-memory retrieval,
    # matching the offline run and the API's .env loading via setdefault.
    env = dict(os.environ)
    env["SUPABASE_URL"] = ""
    env["SUPABASE_SERVICE_KEY"] = ""
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--port", str(chosen), "--log-level", "warning"],
        env=env, cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{chosen}"
    try:
        client = httpx.Client(base_url=base, timeout=60.0)
        deadline = time.time() + 180
        ready = False
        while time.time() < deadline:
            try:
                data = client.get("/health").json()
                if data.get("model_loaded"):
                    print(f"  API ready ({data.get('retrieval_backend')})")
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
        if not ready:
            fail("T10", "API did not report ready within 180s (check /health output)")
            return
        client.post("/resolve", json={"query": "warmup", "with_map": False})

        rng = random.Random(7)
        sample = rng.sample(list(input_texts.items()), 50)
        singles, batch_pairs = sample[:25], sample[25:]

        def key_of(out: dict) -> tuple:
            """Read the (intent, category, decision) triple from either the
            API result shape (predicted_intent) or the predictions.csv shape
            (intent)."""
            intent = out.get("predicted_intent", out.get("intent"))
            category = out.get("predicted_category", out.get("category"))
            return (intent, category, out.get("decision"))

        mism = 0
        for qid, text in singles:
            out = client.post("/resolve",
                              json={"query": text, "with_map": False}).json()
            if key_of(out) != key_of(predictions[qid]):
                mism += 1
                fail("T10", f"/resolve {qid}: API {key_of(out)} != offline "
                            f"{key_of(predictions[qid])}")
            if str(out.get("response", out.get("final_response", ""))) != \
                    str(responses[qid].get("response", "")):
                mism += 1
                fail("T10", f"/resolve {qid}: response text differs from the offline run")
        batch_txt = [text for qid, text in batch_pairs]
        got = client.post("/resolve/batch", json={"queries": batch_txt}).json()["results"]
        for (qid, _text), out in zip(batch_pairs, got):
            if key_of(out) != key_of(predictions[qid]):
                mism += 1
                fail("T10", f"/resolve/batch {qid}: API {key_of(out)} != offline "
                            f"{key_of(predictions[qid])}")
        if mism == 0:
            print(f"  {len(singles)} POST /resolve rows and {len(batch_pairs)} "
                  f"POST /resolve/batch rows equal the offline run")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


# ── main ─────────────────────────────────────────────────────────────────

OUTDIR = "outputs/evaluation"


def main() -> int:
    global OUTDIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="data/eval/evaluation_queries.csv")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--skip-heavy", action="store_true",
                    help="skip T7 re-run and T10 API test (use only to iterate format)")
    args = ap.parse_args()
    OUTDIR = args.outdir

    input_rows = _read_input(args.input)
    pred_path = os.path.join(OUTDIR, "predictions.csv")
    resp_path = os.path.join(OUTDIR, "responses.jsonl")
    trace_path = os.path.join(OUTDIR, "traces.jsonl")
    missing = [p for p in (pred_path, resp_path, trace_path) if not os.path.exists(p)]
    if missing:
        print(f"ERROR: missing output files {missing}; "
              "run scripts/run_evaluation.py first")
        return 1
    header, rows = _read_predictions(pred_path)
    responses = _read_jsonl(resp_path)
    traces = _read_jsonl(trace_path)

    input_texts = run_t1(input_rows)
    predictions = run_t2(header, rows, input_texts)
    run_t3(predictions)
    run_t4(predictions, responses)
    run_t5(input_texts, responses, traces)
    run_t6(responses)
    if not args.skip_heavy:
        run_t7(input_texts, predictions)
        run_t10(input_texts, predictions, responses)
    else:
        head("T7/T10")
        print("  skipped (--skip-heavy)")
    run_t8(input_texts, predictions, responses)
    run_t9(input_texts, predictions)

    print("\n── T1–T10 result table ────────────────────────────────────")
    checks = [f"T{i}" for i in range(1, 11)]
    all_ok = True
    for check in checks:
        issues = len(problems.get(check, []))
        status = "PASS" if issues == 0 else "FAIL"
        all_ok &= issues == 0
        print(f"  {check}  {status}  {issues} issue(s)")
    print("────────────────────────────────────────────────────────────")
    print("ALL CHECKS PASSED" if all_ok else
          f"FAILED: {[c for c in checks if c in problems]}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
