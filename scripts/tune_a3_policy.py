"""Grid-search the A3 routing thresholds on the DEV split and rewrite the policy.

Only `outputs/splits/dev.csv` is used for selection — never TEST. The test split
is evaluated exactly once elsewhere, in the Day-1 evidence scripts.

Objective
---------
Maximise intent accuracy on the items A3 chose to answer, subject to a minimum
answered coverage (so the planner cannot win by answering one easy query).
Ties break on higher coverage, then on a lower clarify rate.

The A1 outputs for every DEV row are cached in
`outputs/cache/a3_dev_a1.json` (keyed by A1 config), because calling the real
A1 `analyze()` 1730 times costs a few minutes and changes nothing between grid
points.

Usage
-----
    python -m scripts.tune_a3_policy            # search + write config/l2_policy.json
    python -m scripts.tune_a3_policy --dry-run  # search + print, do not write
    python -m scripts.tune_a3_policy --refresh  # ignore the A1 output cache
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from nlp.l2.a3 import RoutePlanner  # noqa: E402
from nlp.l2.policy import load_policy, save_policy  # noqa: E402

DEV_CSV = "outputs/splits/dev.csv"
A1_CONFIG = "outputs/models/a1_config.json"
CACHE_PATH = "outputs/cache/a3_dev_a1.json"

GRID = {
    "confidence_high": [0.55, 0.65, 0.75, 0.85, 0.95],
    "confidence_low": [0.20, 0.30, 0.40, 0.50, 0.60],
    "top2_margin_min": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
}


def _config_fingerprint() -> str:
    with open(A1_CONFIG, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_dev_predictions(refresh: bool = False) -> list[dict]:
    """Return [{query, true_intent, true_category, a1}] for the DEV split."""
    fingerprint = _config_fingerprint()
    if not refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cached = json.load(f)
        if cached.get("a1_config_sha") == fingerprint:
            print(f"[tune] using cached A1 outputs ({len(cached['rows'])} dev rows)")
            return cached["rows"]

    from nlp.a1 import IntentAnalyst

    df = pd.read_csv(DEV_CSV)
    a1 = IntentAnalyst()
    rows = []
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False), 1):
        rows.append({
            "query": str(row.instruction),
            "true_intent": str(row.intent),
            "true_category": str(row.category),
            "a1": a1.analyze(str(row.instruction)),
        })
        if i % 250 == 0:
            print(f"[tune] A1 {i}/{len(df)} dev rows ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"a1_config_sha": fingerprint, "rows": rows}, f)
    print(f"[tune] cached A1 outputs -> {CACHE_PATH}")
    return rows


def evaluate(rows: list[dict], base_policy: dict,
             confidence_high: float, confidence_low: float,
             top2_margin_min: float) -> dict:
    """Run A3 over every dev row with one threshold triple."""
    cfg = json.loads(json.dumps(base_policy))  # deep copy
    cfg["thresholds"]["confidence_high"] = confidence_high
    cfg["thresholds"]["confidence_low"] = confidence_low
    cfg["thresholds"]["top2_margin_min"] = top2_margin_min
    planner = RoutePlanner(policy=cfg)

    answered = correct = clarify = escalate = missing = 0
    for row in rows:
        plan = planner.plan(row["query"], row["a1"])
        if plan.route == "answer":
            answered += 1
            correct += plan.intent == row["true_intent"]
        elif plan.route == "clarify":
            clarify += 1
            if "missing_information" in plan.reason_codes:
                missing += 1
        else:
            escalate += 1

    n = len(rows)
    return {
        "confidence_high": confidence_high,
        "confidence_low": confidence_low,
        "top2_margin_min": top2_margin_min,
        "coverage": round(answered / n, 4),
        "answer_accuracy": round(correct / answered, 4) if answered else 0.0,
        "n_answered": answered,
        "clarify_rate": round(clarify / n, 4),
        "escalate_rate": round(escalate / n, 4),
        "missing_info_rate": round(missing / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    base_policy = load_policy()
    rows = load_dev_predictions(refresh=args.refresh)
    min_coverage = float(base_policy["tuning"].get("min_coverage", 0.80))

    results = []
    combos = list(itertools.product(
        GRID["confidence_high"], GRID["confidence_low"], GRID["top2_margin_min"]))
    for conf_high, conf_low, margin_min in combos:
        if conf_low >= conf_high:
            continue
        results.append(evaluate(rows, base_policy, conf_high, conf_low, margin_min))
    print(f"[tune] evaluated {len(results)} threshold triples on {len(rows)} dev rows")

    feasible = [r for r in results if r["coverage"] >= min_coverage]
    pool = feasible or results
    best = sorted(
        pool,
        key=lambda r: (-r["answer_accuracy"], -r["coverage"],
                       r["clarify_rate"] + r["escalate_rate"]),
    )[0]

    baseline = {
        k: base_policy["thresholds"][k]
        for k in ("confidence_high", "confidence_low", "top2_margin_min")
    }
    base_result = evaluate(rows, base_policy, **baseline)
    print(f"[tune] current policy : {base_result}")
    print(f"[tune] selected       : {best}")

    top = sorted(pool, key=lambda r: (-r["answer_accuracy"], -r["coverage"]))[:10]
    policy = json.loads(json.dumps(base_policy))
    policy["thresholds"].update({
        "confidence_high": best["confidence_high"],
        "confidence_low": best["confidence_low"],
        "top2_margin_min": best["top2_margin_min"],
    })
    policy["source"] = "tuned:dev"
    policy["updated"] = time.strftime("%Y-%m-%d")
    tuning = policy.setdefault("tuning", {})
    tuning["n_dev"] = len(rows)
    tuning["grid_size"] = len(results)
    tuning["selected"] = best
    tuning["baseline"] = base_result
    tuning["top"] = top
    tuning["log"] = [
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} dev grid search: "
        f"{len(results)} triples, min_coverage={min_coverage}, selected "
        f"conf_high={best['confidence_high']} conf_low={best['confidence_low']} "
        f"margin_min={best['top2_margin_min']} -> coverage={best['coverage']} "
        f"answered_accuracy={best['answer_accuracy']}"
    ] + tuning.get("log", [])

    if args.dry_run:
        print("[tune] --dry-run: policy not written")
        return
    path = save_policy(policy)
    print(f"[tune] wrote {path}")


if __name__ == "__main__":
    main()
