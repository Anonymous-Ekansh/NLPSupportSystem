#!/usr/bin/env python3
"""Level-2 robustness + decision evaluation on the held-out TEST split.

Everything reported here is computed on `outputs/splits/test.csv` only.  All
thresholds come from `config/l2_policy.json`, which was tuned on DEV — nothing
in this script writes or re-tunes the policy, so no number here is a fit to
TEST.

Sections
--------
1. overall      intent accuracy, intent Macro-F1, category accuracy
2. by_flag      the dataset's own `flags` variation letters (multi-label)
3. by_perturb   seeded synthetic perturbations (typo, swap, abbrev, slang,
                ALL CAPS, missing punctuation, Hinglish)
4. selective    coverage / accuracy-on-ANSWERed / risk-coverage curve
5. stress       ambiguous blends, missing-slot, out-of-scope,
                contradictory-neighbour, weak-evidence
6. claims       false action claims in every ANSWER output (target 0)
7. ablation     Day-1 A1->A2 vs Level-2 on the identical stress cases
8. traces       5 curated full traces (accept, revise, clarify x2, escalate)

Outputs: ``outputs/evidence/robustness.json``, ``l2_decisions.json``,
``trace_examples.json`` (pass ``--sync`` to copy them to web/public).

Usage::

    .venv/bin/python scripts/eval_l2.py                  # ~7 min on CPU
    .venv/bin/python scripts/eval_l2.py --n 200 --n-pert 30 --sync
    .venv/bin/python scripts/eval_l2.py --n 0            # full 3475-row TEST

Rows produced by the perturbation module are marked ``synthetic: true`` and
their expectations are never treated as ground truth — the *label* always
comes from the original TEST row.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                          # noqa: E402
from sklearn.metrics import f1_score                         # noqa: E402

from nlp import perturb as P                                  # noqa: E402
from nlp.l2.a4 import claim_matches                           # noqa: E402
from nlp.l2.contracts import ClarifyOption, RoutePlan, RetrievalSettings  # noqa: E402
from nlp.l2.policy import load_policy                         # noqa: E402
from nlp.pipeline import Level2Pipeline                       # noqa: E402

EVIDENCE_DIR = "outputs/evidence"
LOW_SAMPLE = 30
DECISIONS = ("ANSWER", "CLARIFY", "ESCALATE")


# ═══════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════

def macro_f1(y_true: list, y_pred: list) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def decision_counts(results: list) -> dict:
    counts = {d: 0 for d in DECISIONS}
    for res in results:
        d = res.get("decision")
        if d in counts:
            counts[d] += 1
    return counts


def decision_shares(results: list) -> dict:
    counts = decision_counts(results)
    n = sum(counts.values()) or 1
    return {d: round(c / n, 4) for d, c in counts.items()}


def prediction_metrics(results: list) -> dict:
    """Intent/category metrics + decision distribution for one group."""
    gold = [r["gold_intent"] for r in results]
    pred = [r["predicted_intent"] or "" for r in results]
    gold_c = [r["gold_category"] for r in results]
    pred_c = [r["predicted_category"] or "" for r in results]
    return {
        "n": len(results),
        "low_sample": len(results) < LOW_SAMPLE,
        "intent_accuracy": round(accuracy(gold, pred), 4),
        "intent_macro_f1": round(macro_f1(gold, pred), 4),
        "category_accuracy": round(accuracy(gold_c, pred_c), 4),
        "decisions": decision_counts(results),
        "decision_shares": decision_shares(results),
    }


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional sample by intent; n=0 means the whole split."""
    if n <= 0 or n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = random.Random(seed)
    picks: list[int] = []
    for intent, group in df.groupby("intent"):
        share = len(group) / len(df)
        take = max(1, round(share * n))
        idx = list(group.index)
        rng.shuffle(idx)
        picks.extend(idx[:take])
    rng.shuffle(picks)
    return df.loc[picks[:n]].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline runs
# ═══════════════════════════════════════════════════════════════════════

def run_l2(pipe: Level2Pipeline, queries: list[str], label: str) -> list[dict]:
    out = []
    t0 = time.perf_counter()
    for i, q in enumerate(queries, start=1):
        res = pipe.resolve(q, with_map=False)
        res["query"] = q
        out.append(res)
        if i % 50 == 0 or i == len(queries):
            print(f"    [{label}] {i}/{len(queries)} "
                  f"({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    return out


def run_baseline(pipe: Level2Pipeline, queries: list[str]) -> list[dict]:
    return [pipe.resolve(q, baseline=True) for q in queries]


def attach_gold(results: list[dict], rows: list[dict]) -> list[dict]:
    for res, row in zip(results, rows):
        res["gold_intent"] = row["intent"]
        res["gold_category"] = row["category"]
    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. Selective prediction + risk-coverage
# ═══════════════════════════════════════════════════════════════════════

def selective_prediction(results: list[dict]) -> dict:
    """Operate as the system does (ANSWER is the only committed decision)."""
    answered = [r for r in results if r.get("decision") == "ANSWER"]
    n = len(results) or 1
    correct = sum(r["predicted_intent"] == r["gold_intent"] for r in answered)
    return {
        "coverage": round(len(answered) / n, 4),
        "n_answered": len(answered),
        "accuracy_on_answered": round(correct / len(answered), 4) if answered else None,
        "answered_with_correct_intent": correct,
        "accuracy_if_all_answered": round(
            sum(r["predicted_intent"] == r["gold_intent"] for r in results) / n, 4),
    }


def risk_coverage(results: list[dict], steps: int = 20) -> list[dict]:
    """Confidence-threshold sweep: abstain on the least confident queries."""
    ranked = sorted(results, key=lambda r: -float(r.get("confidence") or 0.0))
    n = len(ranked)
    points = []
    for k in range(1, steps + 1):
        coverage = k / steps
        cut = max(1, int(round(coverage * n)))
        subset = ranked[:cut]
        acc = sum(r["predicted_intent"] == r["gold_intent"]
                  for r in subset) / len(subset)
        points.append({
            "coverage": round(cut / n, 4),
            "accuracy": round(acc, 4),
            "n_answered": cut,
            "confidence_floor": round(float(subset[-1].get("confidence") or 0.0), 4),
        })
    return points


# ═══════════════════════════════════════════════════════════════════════
# 5. Stress sets
# ═══════════════════════════════════════════════════════════════════════

MISSING_SLOT_TEMPLATES = (
    "I need a refund but I cannot find my order number",
    "I want to cancel my order but I lost the order id",
    "where is my refund? I do not have my order number",
    "please cancel my order, I cannot find my order number anywhere",
    "I need to change my shipping address but I lost my order number",
    "my refund has not arrived and I cannot find my order number",
    "can you tell me my refund status, I lost the order number",
    "I want my money back but I cannot find the order number",
)

OUT_OF_SCOPE_TEMPLATES = (
    "what is the weather in Paris tomorrow",
    "write me a poem about cats",
    "tell me a joke",
    "who won the world cup in 2018",
    "translate this sentence into French",
    "give me a recipe for paneer butter masala",
    "what is the capital of Australia",
    "explain quantum entanglement simply",
    "book me a flight to Tokyo next Friday",
    "what should I do about a persistent headache",
)


def build_ambiguous(df: pd.DataFrame, n: int, seed: int) -> list[dict]:
    """Two real TEST requests from *different categories* in one message.

    A confident single-intent ANSWER is the failure mode; asking or handing
    off is correct.  Labels stay attached for traceability but are not used
    as ground truth (no dataset row is a blend).
    """
    rng = random.Random(seed + 101)
    rows = df.sample(min(len(df), 400), random_state=seed).to_dict("records")
    cases, used = [], set()
    for _ in range(n * 12):
        a, b = rng.choice(rows), rng.choice(rows)
        if a["category"] == b["category"] or a["instruction"] == b["instruction"]:
            continue
        if a["instruction"] in used or b["instruction"] in used:
            continue
        used.update({a["instruction"], b["instruction"]})
        cases.append({
            "id": f"AMB{len(cases) + 1:02d}",
            "query": f"{str(a['instruction']).strip().rstrip('?')}, and "
                     f"{str(b['instruction']).strip()[0].lower()}"
                     f"{str(b['instruction']).strip()[1:]}",
            "components": [a["intent"], b["intent"]],
            "categories": [a["category"], b["category"]],
            "expected": ["CLARIFY", "ESCALATE"],
        })
        if len(cases) >= n:
            break
    return cases


def build_missing_slot() -> list[dict]:
    return [{"id": f"MIS{i + 1:02d}", "query": q,
             "expected": ["CLARIFY", "ESCALATE"], "preferred": "CLARIFY"}
            for i, q in enumerate(MISSING_SLOT_TEMPLATES)]


def build_out_of_scope() -> list[dict]:
    return [{"id": f"OOS{i + 1:02d}", "query": q, "expected": ["ESCALATE"]}
            for i, q in enumerate(OUT_OF_SCOPE_TEMPLATES)]


def build_weak_evidence(df: pd.DataFrame, results: list[dict],
                        n: int) -> list[dict]:
    """Real TEST rows the system itself rates as uncertain.

    No synthetic text: these are held-out queries where A1's confidence or
    neighbour similarity is weak, so the honest expectation is abstention.
    """
    weak = [(res, row) for res, row in zip(results, df.to_dict("records"))
            if res["a1"].get("needs_clarification")]
    weak.sort(key=lambda pair: float(pair[0]["a1"].get("confidence") or 0.0))
    return [{
        "id": f"WK{i + 1:03d}",
        "query": row["instruction"],
        "gold_intent": row["intent"],
        "gold_category": row["category"],
        "confidence": res["a1"].get("confidence"),
        "top1_similarity": res["a1"].get("top1_similarity"),
        "expected": ["CLARIFY", "ESCALATE"],
    } for i, (res, row) in enumerate(weak[:n])]


CONTRADICTION_QUERY = "I need a refund for my order"
CONTRADICTION_EVIDENCE = (
    ("can i get a refund please", "You can request a refund within 30 days of purchase.", "get_refund"),
    ("how do i recover my password", "Click reset password and follow the link we send you.", "recover_password"),
    ("cancel my order", "Orders can be cancelled before they ship.", "cancel_order"),
)


def build_contradictory(pipe: Level2Pipeline) -> list[dict]:
    """Force conflicting retrieved evidence and record what A4/A5 do.

    Retrieval is deterministic in-memory nearest-neighbour, so a genuine
    contradiction cannot be requested through the public API.  This drives
    A4/A5 directly with a constructed evidence set and is reported as a
    *probe*, not as a measurement of retrieval quality.
    """
    from nlp.a5 import coordinate
    from nlp.l2.a4 import audit

    cases = []
    for i, keep in enumerate((0, 1)):
        query = CONTRADICTION_QUERY
        a1 = pipe.a1.analyze(query)
        plan = pipe.planner.plan(query, a1)
        a2 = {
            "response": CONTRADICTION_EVIDENCE[keep][1],
            "supporting_examples": [
                {"instruction": ins, "response": resp, "similarity": 0.9,
                 "intent": tag}
                for ins, resp, tag in CONTRADICTION_EVIDENCE
            ],
            "used_intents": [tag for _, _, tag in CONTRADICTION_EVIDENCE],
            "retrieval_score": 0.9, "low_confidence": False,
            "self_check": {"check_passed": True},
        }
        audit_result = audit(query, a1, plan, a2)
        decision, steps = coordinate(query, a1, plan, a2, audit_result, {})
        cases.append({
            "id": f"CON{i + 1:02d}",
            "query": query,
            "probe": "constructed contradictory evidence",
            "expected": ["ESCALATE"],
            "a1": a1, "a3": plan.to_dict(),
            "a4": audit_result.to_dict(),
            "decision": decision.decision,
            "response": decision.response,
            "trace": [s.to_dict() for s in steps],
        })
    return cases


def score_stress(name: str, cases: list[dict],
                 l2_by_query: dict) -> dict:
    """Correct-routing vs wrong-ANSWER share for one stress set."""
    rows, wrong = [], 0
    for case in cases:
        if "decision" in case:                     # pre-computed probe
            decision = case["decision"]
            query = case["query"]
        else:
            res = l2_by_query.get(case["query"])
            decision = (res or {}).get("decision")
            query = case["query"]
        ok = decision in case["expected"]
        if decision == "ANSWER":
            wrong += 1
        rows.append({"id": case["id"], "query": query, "decision": decision,
                     "correct": ok, "expected": case["expected"]})
    n = len(rows) or 1
    return {
        "n": len(rows),
        "low_sample": len(rows) < LOW_SAMPLE,
        "correctly_routed": sum(r["correct"] for r in rows),
        "correctly_routed_share": round(sum(r["correct"] for r in rows) / n, 4),
        "wrongly_answered": wrong,
        "wrongly_answered_share": round(wrong / n, 4),
        "decisions": dict(Counter(r["decision"] for r in rows)),
        "cases": rows,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6/7. Claim guard + Day-1 ablation
# ═══════════════════════════════════════════════════════════════════════

def count_false_claims(results: list[dict]) -> dict:
    """Action-completion claims anywhere in an ANSWER output (target 0)."""
    checked, hits = 0, []
    for res in results:
        if res.get("decision") != "ANSWER":
            continue
        checked += 1
        text = res.get("final_response") or ""
        for match in claim_matches(text):
            hits.append({"query": res.get("query"),
                         "match": match.group(0),
                         "response": text[:240]})
    return {"answers_checked": checked, "false_action_claims": len(hits),
            "target": 0, "passed": len(hits) == 0, "examples": hits[:10]}


def baseline_stress_profile(pipe: Level2Pipeline, queries: list[str]) -> dict:
    """What Day-1 does on the same cases.

    Day 1 has no abstention layer: it always returns a response.  So its
    `answered_share` is 1.0 by construction — the informative columns are
    whether that response happened to ask something useful and whether it
    contained an action claim.
    """
    results = run_baseline(pipe, queries)
    n = len(results) or 1
    asks = sum("?" in (r.get("final_response") or "") for r in results)
    claims = sum(bool(claim_matches(r.get("final_response") or ""))
                 for r in results)
    return {
        "n": len(results),
        "answered_share": 1.0,
        "asks_a_question_share": round(asks / n, 4),
        "action_claim_share": round(claims / n, 4),
        "note": "Day 1 has no decision layer, so it answers every query; "
                "this is a property of the architecture, not a measured win.",
    }


# ═══════════════════════════════════════════════════════════════════════
# 8. Curated traces
# ═══════════════════════════════════════════════════════════════════════

def curl_traces(clean_results: list[dict], extra: list[dict]) -> list[dict]:
    """Five traces covering accept, revise, clarify (missing + ambiguous),
    escalate."""
    wanted = [
        ("answer_accept", "A4 accepted the retrieved response",
         lambda r: (r.get("decision") == "ANSWER"
                    and (r.get("a4") or {}).get("verdict") == "accept")),
        ("answer_revise", "A4 rewrote the candidate before serving it",
         lambda r: (r.get("decision") == "ANSWER"
                    and (r.get("a4") or {}).get("verdict") == "revise")),
    ]
    picked = []
    for label, why, pred in wanted:
        found = next((r for r in clean_results if pred(r)), None)
        if found:
            picked.append((label, why, found))

    for res in extra:
        if res.get("decision") == "CLARIFY" and (res.get("a3") or {}).get("missing_slots"):
            picked.append(("clarify_missing_slot",
                           "essential information is missing: A5 asks for it",
                           res))
            break
    for res in extra:
        if (res.get("decision") == "CLARIFY"
                and (res.get("a3") or {}).get("clarify_options")):
            picked.append(("clarify_ambiguous",
                           "two intents compete: A5 asks which one is meant", res))
            break
    for res in extra:
        if res.get("decision") == "ESCALATE":
            picked.append(("escalate_handoff",
                           "unsupported or sensitive: A5 hands off to a human",
                           res))
            break

    from nlp.a5 import intent_label

    examples = []
    for label, why, res in picked[:5]:
        examples.append({
            "label": label,
            "why": why,
            "query": res.get("query"),
            "decision": res.get("decision"),
            "response": res.get("final_response"),
            "predicted_intent": res.get("predicted_intent"),
            "human_intent": intent_label(res.get("predicted_intent") or ""),
            "predicted_category": res.get("predicted_category"),
            "reason_codes": res.get("reason_codes", []),
            "total_latency_ms": res.get("total_latency_ms"),
            "trace": res.get("trace", []),
        })
    return examples


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500,
                    help="clean TEST rows (0 = whole split)")
    ap.add_argument("--n-pert", dest="n_pert", type=int, default=60,
                    help="rows per synthetic perturbation kind")
    ap.add_argument("--n-weak", dest="n_weak", type=int, default=25,
                    help="weak-evidence cases")
    ap.add_argument("--n-ambiguous", dest="n_ambiguous", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", dest="out_dir", default=EVIDENCE_DIR,
                    help="where to write the evidence JSON")
    ap.add_argument("--sync", action="store_true",
                    help="copy evidence to web/public via scripts/sync_evidence.py")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    os.makedirs(args.out_dir, exist_ok=True)

    policy = load_policy()
    thresholds = policy.get("thresholds", {})
    df = pd.read_csv(f"outputs/splits/{args.split}.csv")
    print(f"[eval] {args.split}: {len(df)} rows; policy "
          f"{policy.get('version')} (tuned on DEV)")

    pipe = Level2Pipeline()
    print("[eval] models loaded")

    # ── 1. clean held-out sample ─────────────────────────────────────
    clean = stratified_sample(df, args.n, args.seed)
    print(f"[eval] clean sample: {len(clean)} rows")
    clean_rows = clean.to_dict("records")
    clean_results = attach_gold(
        run_l2(pipe, [str(r["instruction"]) for r in clean_rows], "clean"),
        clean_rows)

    overall = prediction_metrics(clean_results)

    day1_ref = {}
    if os.path.exists("outputs/metrics/test_metrics.json"):
        with open("outputs/metrics/test_metrics.json") as fh:
            ref = json.load(fh)
        day1_ref = {k: ref.get(k) for k in
                    ("n_samples", "intent_accuracy", "intent_f1_macro",
                     "category_accuracy")}

    overall["day1_reference"] = day1_ref
    overall["day1_note"] = (
        "Day-1 figures are computed over the full 3475-row TEST split by "
        "nlp/evaluate.py; these are over the seeded sample, so small "
        "differences are sampling noise. Run with --n 0 to compare exactly.")
    if day1_ref.get("intent_accuracy") is not None:
        overall["intent_accuracy_delta_vs_day1"] = round(
            overall["intent_accuracy"] - day1_ref["intent_accuracy"], 4)
        overall["matches_day1_within_0.01"] = abs(
            overall["intent_accuracy_delta_vs_day1"]) <= 0.01

    # ── 2. by dataset flag group ─────────────────────────────────────
    letter_rows: dict[str, list[dict]] = defaultdict(list)
    for res, row in zip(clean_results, clean_rows):
        for letter in P.flag_letters(row["flags"]):
            if letter in P.NON_VARIATION_FLAGS:
                continue
            letter_rows[letter].append(res)

    by_flag = []
    for letter in sorted(letter_rows, key=lambda k: -len(letter_rows[k])):
        block = prediction_metrics(letter_rows[letter])
        meta = P.flag_group(letter)
        by_flag.append({
            "group": letter,
            "label": meta["label"],
            "meaning": meta["meaning"],
            "multi_label": True,
            **block,
        })
    print(f"[eval] flag groups: {len(by_flag)}")

    # ── 3. synthetic perturbations ───────────────────────────────────
    pert_rows = stratified_sample(clean, min(args.n_pert, len(clean)),
                                  args.seed + 1).to_dict("records")
    # How often each transform can even apply, measured over the whole split.
    transform_coverage = {}
    for kind in P.SYNTHETIC_KINDS:
        changed = sum(P.is_changed(str(q), P.perturb(str(q), kind, args.seed))
                      for q in df["instruction"].astype(str))
        transform_coverage[kind] = round(changed / len(df), 4)

    by_perturb = []
    for kind in P.SYNTHETIC_KINDS:
        applicable = [r for r in pert_rows
                      if P.is_changed(str(r["instruction"]),
                                      P.perturb(str(r["instruction"]), kind,
                                                args.seed))]
        if not applicable:
            by_perturb.append({
                "group": kind, "n": 0, "low_sample": True,
                "transform_coverage": transform_coverage[kind],
                "note": "no sampled row is affected by this transform",
            })
            continue
        variants = [{"instruction": P.perturb(str(r["instruction"]), kind,
                                              args.seed),
                     "intent": r["intent"], "category": r["category"]}
                    for r in applicable]
        results = attach_gold(
            run_l2(pipe, [v["instruction"] for v in variants], kind), variants)
        block = prediction_metrics(results)
        by_perturb.append({
            "group": kind,
            "synthetic": True,
            "transform_coverage": transform_coverage[kind],
            "note": "labels inherited from the original TEST row",
            **block,
        })
        print(f"    {kind}: acc {block['intent_accuracy']} "
              f"(clean {overall['intent_accuracy']})")

    # ── 4. selective prediction ──────────────────────────────────────
    selective = selective_prediction(clean_results)
    selective["risk_coverage_curve"] = risk_coverage(clean_results)

    # ── 5. stress sets ───────────────────────────────────────────────
    print("[eval] stress sets")
    ambiguous_cases = build_ambiguous(df, args.n_ambiguous, args.seed)
    missing_cases = build_missing_slot()
    oos_cases = build_out_of_scope()
    weak_cases = build_weak_evidence(df, clean_results, args.n_weak)

    text_cases = ambiguous_cases + missing_cases + oos_cases
    stress_l2 = run_l2(pipe, [c["query"] for c in text_cases], "stress")
    by_query = {c["query"]: res for c, res in zip(text_cases, stress_l2)}

    weak_l2 = run_l2(pipe, [c["query"] for c in weak_cases], "weak")
    by_query.update({c["query"]: res for c, res in zip(weak_cases, weak_l2)})

    contradictory_cases = build_contradictory(pipe)

    stress_sets = {
        "ambiguous": score_stress("ambiguous", ambiguous_cases, by_query),
        "missing_slot": score_stress("missing_slot", missing_cases, by_query),
        "out_of_scope": score_stress("out_of_scope", oos_cases, by_query),
        "weak_evidence": score_stress("weak_evidence", weak_cases, by_query),
        "contradictory_neighbour": score_stress("contradictory_neighbour",
                                                contradictory_cases, by_query),
    }
    stress_sets["missing_slot"]["preferred_decision"] = "CLARIFY"
    stress_sets["contradictory_neighbour"]["probe_note"] = (
        "Constructed conflicting evidence driven through A4/A5 directly; it "
        "tests the audit/coordination layer, not retrieval.")
    stress_sets["ambiguous"]["assumption"] = (
        "A blend of two requests from different categories has no single "
        "correct intent; CLARIFY or ESCALATE counts as correct, a confident "
        "single-intent ANSWER counts as wrong.")

    # ── 6. false action claims ───────────────────────────────────────
    all_answers = (clean_results
                   + [r for r in stress_l2]
                   + [r for r in weak_l2])
    claims = count_false_claims(all_answers)

    # ── 7. Day-1 vs Level-2 ablation ─────────────────────────────────
    print("[eval] Day-1 ablation on identical stress cases")
    ablation = {}
    for name, cases in (("ambiguous", ambiguous_cases),
                        ("missing_slot", missing_cases),
                        ("out_of_scope", oos_cases),
                        ("weak_evidence", weak_cases)):
        queries = [c["query"] for c in cases]
        l2_block = stress_sets[name]
        day1_block = baseline_stress_profile(pipe, queries)
        ablation[name] = {
            "n": len(queries),
            "level2_wrong_answer_share": l2_block["wrongly_answered_share"],
            "level2_correct_routing_share": l2_block["correctly_routed_share"],
            "day1_answered_share": day1_block["answered_share"],
            "day1_asks_question_share": day1_block["asks_a_question_share"],
            "day1_action_claim_share": day1_block["action_claim_share"],
        }

    # ── 8. curated traces ────────────────────────────────────────────
    extra = []
    for case in missing_cases[:1] + oos_cases[:1]:
        if case["query"] in by_query:
            extra.append(by_query[case["query"]])
    ambiguous_bare = pipe.resolve("refund", with_map=False)
    ambiguous_bare["query"] = "refund"
    extra.insert(1, ambiguous_bare)
    traces = curl_traces(clean_results, extra)

    # ── write robustness.json ────────────────────────────────────────
    runtime = time.perf_counter() - started
    robustness = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split": args.split,
            "n_split": len(df),
            "seed": args.seed,
            "clean_sample_n": len(clean_results),
            "perturb_sample_n": len(pert_rows),
            "low_sample_threshold": LOW_SAMPLE,
            "policy": {"version": policy.get("version"),
                       "path": "config/l2_policy.json",
                       "tuned_on": "DEV (validation)"},
            "thresholds": thresholds,
            "runtime_s": round(runtime, 1),
            "synthetic_note": (
                "Synthetic rows come from nlp/perturb.py (seeded, "
                "deterministic). Their labels are inherited from the original "
                "TEST row; expectations for stress sets are stated in "
                "'expected' and are our own definition, not dataset truth."),
        },
        "overall": overall,
        "by_flag": by_flag,
        "by_perturbation": by_perturb,
        "selective_prediction": selective,
        "stress_sets": stress_sets,
        "false_action_claims": claims,
        "ablation_day1_vs_l2": ablation,
    }
    path = os.path.join(args.out_dir, "robustness.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(robustness, fh, indent=2, ensure_ascii=False)

    # ── write l2_decisions.json ──────────────────────────────────────
    by_intent = defaultdict(list)
    by_category = defaultdict(list)
    codes = Counter()
    for res in clean_results:
        by_intent[res["gold_intent"]].append(res)
        by_category[res["gold_category"]].append(res)
        codes.update(res.get("reason_codes", []))

    l2_decisions = {
        "meta": {
            "generated_at": robustness["meta"]["generated_at"],
            "split": args.split,
            "n": len(clean_results),
            "policy_version": policy.get("version"),
        },
        "decisions": decision_counts(clean_results),
        "decision_shares": decision_shares(clean_results),
        "reason_codes": dict(codes.most_common()),
        "by_intent": {
            intent: {
                "n": len(rows),
                "low_sample": len(rows) < LOW_SAMPLE,
                "decision_shares": decision_shares(rows),
                "decisions": decision_counts(rows),
                "intent_accuracy": round(accuracy(
                    [r["gold_intent"] for r in rows],
                    [r["predicted_intent"] or "" for r in rows]), 4),
            } for intent, rows in sorted(by_intent.items())
        },
        "by_category": {
            cat: {
                "n": len(rows),
                "low_sample": len(rows) < LOW_SAMPLE,
                "decision_shares": decision_shares(rows),
                "decisions": decision_counts(rows),
            } for cat, rows in sorted(by_category.items())
        },
        "selective_prediction": selective,
    }
    with open(os.path.join(args.out_dir, "l2_decisions.json"), "w",
              encoding="utf-8") as fh:
        json.dump(l2_decisions, fh, indent=2, ensure_ascii=False)

    # ── write trace_examples.json ────────────────────────────────────
    with open(os.path.join(args.out_dir, "trace_examples.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "meta": {"generated_at": robustness["meta"]["generated_at"],
                     "split": args.split,
                     "note": "Real traces from the Level-2 pipeline."},
            "examples": traces,
        }, fh, indent=2, ensure_ascii=False)

    # ── console summary ──────────────────────────────────────────────
    print("\n" + "=" * 66)
    print(f"intent accuracy   {overall['intent_accuracy']}  "
          f"(macro-F1 {overall['intent_macro_f1']})")
    print(f"category accuracy {overall['category_accuracy']}")
    print(f"decisions         {decision_counts(clean_results)}")
    print(f"coverage          {selective['coverage']}  "
          f"acc-on-ANSWER {selective['accuracy_on_answered']}")
    for name, block in stress_sets.items():
        print(f"  {name:24s} n={block['n']:3d} "
              f"correct={block['correctly_routed_share']:.2f} "
              f"wrong-ANSWER={block['wrongly_answered_share']:.2f}")
    print(f"false action claims {claims['false_action_claims']} "
          f"/ {claims['answers_checked']} answers")
    print(f"runtime {runtime:.0f}s")
    print("=" * 66)
    print(f"wrote {path}")
    print(f"wrote {os.path.join(args.out_dir, 'l2_decisions.json')}")
    print(f"wrote {os.path.join(args.out_dir, 'trace_examples.json')}")

    if args.sync:
        sys.path.insert(0, "scripts")
        import sync_evidence
        sync_evidence.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
