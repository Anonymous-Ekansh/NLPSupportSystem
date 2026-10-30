#!/usr/bin/env python3
"""Evaluate Agent A2 on TEST using real A1 outputs.

Metrics (no LLM):
  - Relevance: cosine(emb(returned_response), emb(reference_response)) + ROUGE-L F1
  - Intent consistency: share where source example intent == true intent
  - Placeholder-safe rate
  - Self-check rejection rate
  - Compare sparse/dense/hybrid ± A1 routing + random baseline
  - Slice by A1-misclassified and low-confidence

Usage:  python scripts/eval_a2.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.normalize import normalize
from nlp.embed import embed, embed_split
from nlp.a1 import IntentAnalyst
from nlp.a2 import A2Builder

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
EVIDENCE_DIR = "outputs/evidence"
MODEL_DIR = "outputs/models"

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
# ROUGE-L F1 (pure Python, no external lib)
# ═══════════════════════════════════════════════════════════════════════

def _lcs_length(x: list, y: list) -> int:
    """Length of longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 score between two strings (word-level)."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(hyp_tokens)
    rec = lcs / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Load config
    with open(os.path.join(MODEL_DIR, "a1_config.json")) as f:
        a1_config = json.load(f)
    embed_model = a1_config.get("embed_model", "BAAI/bge-small-en-v1.5")

    print("=" * 70)
    print("A2 EVALUATION")
    print("=" * 70)

    # Load A1
    print("\n  Loading A1 …")
    a1 = IntentAnalyst()

    # Build A2
    print("  Building A2 …")
    a2 = A2Builder(embed_model=embed_model).build()

    # Load test data
    test_df = pd.read_csv("outputs/splits/test.csv")
    dev_df = pd.read_csv("outputs/splits/dev.csv")
    print(f"  TEST: {len(test_df)} rows")
    print(f"  DEV:  {len(dev_df)} rows")

    # Pre-run A1 on all test queries
    print("\n  Running A1 on all TEST queries …")
    a1_outputs = []
    for i, row in test_df.iterrows():
        a1_out = a1.analyze(str(row["instruction"]))
        a1_outputs.append(a1_out)
        if (i + 1) % 500 == 0:
            print(f"    A1: {i + 1}/{len(test_df)}")

    a1_preds = [o["intent"] for o in a1_outputs]
    a1_correct = np.array([
        a1_outputs[i]["intent"] == test_df.iloc[i]["intent"]
        for i in range(len(test_df))
    ])
    a1_low_conf = np.array([o["needs_clarification"] for o in a1_outputs])

    print(f"    A1 accuracy: {a1_correct.mean():.4f}")
    print(f"    A1 low-confidence: {a1_low_conf.mean():.2%}")

    # ── Evaluate each mode ────────────────────────────────────────────

    modes = ["sparse", "dense", "hybrid"]
    routings = [True, False]  # True = with A1 routing, False = global
    configs = []
    for mode in modes:
        for routed in routings:
            configs.append((mode, routed))
    # Add random baseline
    configs.append(("random", True))

    all_results = {}

    for mode, routed in configs:
        label = f"{mode}_{'routed' if routed else 'global'}"
        print(f"\n  Evaluating: {label} …")

        cosine_sims = []
        rouge_ls = []
        intent_consistent = []
        placeholder_safe = []
        check_rejections = []
        check_passed_list = []

        for i, row in test_df.iterrows():
            query = str(row["instruction"])
            ref_response = str(row["response"])
            true_intent = row["intent"]

            if mode == "random":
                # Random response from true intent
                same_intent = a2.train_df[a2.train_df["intent"] == true_intent]
                if len(same_intent) > 0:
                    sampled = same_intent.sample(1, random_state=SEED + i)
                    resp = str(sampled.iloc[0]["response"])
                    src_intent = str(sampled.iloc[0]["intent"])
                else:
                    resp = ref_response
                    src_intent = true_intent

                cosine_sims.append(0.0)  # placeholder, compute below
                rouge_ls.append(rouge_l_f1(resp, ref_response))
                intent_consistent.append(src_intent == true_intent)
                placeholder_safe.append("{{" not in resp)
                check_rejections.append(0)
                check_passed_list.append(True)
                continue

            if routed:
                result = a2.respond(query, a1_outputs[i], mode=mode)
            else:
                result = a2.respond_global(query, mode=mode)

            resp = result["response"]

            # ROUGE-L
            rouge_ls.append(rouge_l_f1(resp, ref_response))

            # Intent consistency: source example intent == true intent
            used = result.get("used_intents", [])
            intent_consistent.append(true_intent in used)

            # Placeholder safe
            placeholder_safe.append("{{" not in resp)

            if "self_check" in result:
                check_rejections.append(result["self_check"].get("candidates_rejected", 0))
                check_passed_list.append(result["self_check"].get("check_passed", True))
            else:
                check_rejections.append(0)
                check_passed_list.append(True)

            # cosine sim placeholder
            cosine_sims.append(0.0)

            if (i + 1) % 500 == 0:
                print(f"    {label}: {i + 1}/{len(test_df)}")

        # Compute cosine similarities in batch
        print(f"    Computing response embeddings for {label} …")
        # Re-run to collect responses for embedding
        returned_responses = []
        for i, row in test_df.iterrows():
            query = str(row["instruction"])
            if mode == "random":
                same_intent = a2.train_df[a2.train_df["intent"] == row["intent"]]
                if len(same_intent) > 0:
                    sampled = same_intent.sample(1, random_state=SEED + i)
                    returned_responses.append(str(sampled.iloc[0]["response"]))
                else:
                    returned_responses.append(str(row["response"]))
            elif routed:
                result = a2.respond(query, a1_outputs[i], mode=mode)
                returned_responses.append(result["response"])
            else:
                result = a2.respond_global(query, mode=mode)
                returned_responses.append(result["response"])

        ref_responses = [str(r) for r in test_df["response"].values]

        # Embed returned and reference responses
        ret_emb = embed([normalize(r) for r in returned_responses],
                       model=embed_model, cache_tag=f"a2_ret_{label}")
        ref_emb = embed([normalize(r) for r in ref_responses],
                       model=embed_model, cache_tag="a2_ref")

        cosine_sims = (ret_emb * ref_emb).sum(axis=1).tolist()

        # Aggregate metrics
        metrics = {
            "cosine_sim": round(float(np.mean(cosine_sims)), 4),
            "rouge_l_f1": round(float(np.mean(rouge_ls)), 4),
            "intent_consistency": round(float(np.mean(intent_consistent)), 4),
            "placeholder_safe_rate": round(float(np.mean(placeholder_safe)), 4),
            "check_rejection_rate": round(
                float(np.mean([r > 0 for r in check_rejections])), 4),
            "check_passed_rate": round(float(np.mean(check_passed_list)), 4),
        }

        # Slices
        slices = {}
        if mode != "random":
            # A1-misclassified
            mis_mask = ~a1_correct
            if mis_mask.sum() > 0:
                slices["a1_misclassified"] = {
                    "n": int(mis_mask.sum()),
                    "cosine_sim": round(float(np.array(cosine_sims)[mis_mask].mean()), 4),
                    "rouge_l_f1": round(float(np.array(rouge_ls)[mis_mask].mean()), 4),
                    "intent_consistency": round(
                        float(np.array(intent_consistent)[mis_mask].mean()), 4),
                }

            # Low confidence
            if a1_low_conf.sum() > 0:
                slices["low_confidence"] = {
                    "n": int(a1_low_conf.sum()),
                    "cosine_sim": round(
                        float(np.array(cosine_sims)[a1_low_conf].mean()), 4),
                    "rouge_l_f1": round(
                        float(np.array(rouge_ls)[a1_low_conf].mean()), 4),
                    "intent_consistency": round(
                        float(np.array(intent_consistent)[a1_low_conf].mean()), 4),
                }

        metrics["slices"] = slices
        all_results[label] = metrics
        print(f"    → cosine={metrics['cosine_sim']:.4f}  "
              f"rouge={metrics['rouge_l_f1']:.4f}  "
              f"intent_cons={metrics['intent_consistency']:.4f}")

    # ── Choose served mode on DEV ─────────────────────────────────────

    print("\n" + "=" * 70)
    print("CHOOSING SERVED MODE ON DEV")
    print("=" * 70)

    dev_scores = {}
    for mode in modes:
        print(f"\n  DEV: {mode}_routed …")
        rouge_sum = 0.0
        for i, row in dev_df.iterrows():
            query = str(row["instruction"])
            ref_response = str(row["response"])
            a1_out = a1.analyze(query)
            result = a2.respond(query, a1_out, mode=mode)
            rouge_sum += rouge_l_f1(result["response"], ref_response)
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(dev_df)}")
        dev_scores[mode] = round(rouge_sum / len(dev_df), 4)
        print(f"    → DEV ROUGE-L: {dev_scores[mode]:.4f}")

    served_mode = max(dev_scores, key=dev_scores.get)
    print(f"\n  ★ Served mode: {served_mode} (DEV ROUGE-L = {dev_scores[served_mode]:.4f})")

    # ── Save 30 random samples ────────────────────────────────────────

    sample_idx = np.random.choice(len(test_df), 30, replace=False)
    samples = []
    for i in sample_idx:
        row = test_df.iloc[i]
        query = str(row["instruction"])
        result = a2.respond(query, a1_outputs[i], mode=served_mode)
        samples.append({
            "query": query,
            "true_intent": row["intent"],
            "true_category": row["category"],
            "a1_intent": a1_outputs[i]["intent"],
            "a1_confidence": a1_outputs[i]["confidence"],
            "returned_response": result["response"][:300],
            "retrieval_score": result["retrieval_score"],
            "check_passed": result["self_check"]["check_passed"],
            "reference_response": str(row["response"])[:300],
        })

    samples_df = pd.DataFrame(samples)
    samples_df.to_csv("outputs/a2_samples.csv", index=False)
    print(f"\n  Saved 30 samples to outputs/a2_samples.csv")

    # ── Build evidence JSON ───────────────────────────────────────────

    evidence = {
        "served_mode": served_mode,
        "dev_rouge_scores": dev_scores,
        "test_results": all_results,
        "n_test": len(test_df),
        "a1_accuracy_on_test": round(float(a1_correct.mean()), 4),
        "a1_low_conf_rate": round(float(a1_low_conf.mean()), 4),
    }

    with open(os.path.join(EVIDENCE_DIR, "a2.json"), "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════════
    # Build a2_results.md
    # ═══════════════════════════════════════════════════════════════════

    md = []
    md.append("# A2 Response Specialist — Evaluation Results\n")
    md.append(f"**Served mode:** `{served_mode}` with A1 routing\n")
    md.append(f"**Test set:** {len(test_df)} queries\n")
    md.append(f"**A1 accuracy on TEST:** {a1_correct.mean():.4f}\n")
    md.append(f"**A1 low-confidence rate:** {a1_low_conf.mean():.2%}\n")

    # Comparison table
    md.append("\n## Retrieval Mode Comparison (TEST)\n")
    md.append("| Mode | Cosine Sim | ROUGE-L F1 | Intent Cons. | PH-Safe | "
              "Check-Pass | Check-Reject |")
    md.append("|------|-----------|-----------|-------------|---------|"
              "-----------|-------------|")
    for label, m in all_results.items():
        star = " ★" if label == f"{served_mode}_routed" else ""
        md.append(
            f"| {label}{star} | {m['cosine_sim']:.4f} | {m['rouge_l_f1']:.4f} | "
            f"{m['intent_consistency']:.4f} | {m['placeholder_safe_rate']:.4f} | "
            f"{m['check_passed_rate']:.4f} | {m['check_rejection_rate']:.4f} |"
        )

    # Commentary
    md.append("\n## Analysis\n")

    # Compare routed vs global for each mode
    for mode in modes:
        r = all_results.get(f"{mode}_routed", {})
        g = all_results.get(f"{mode}_global", {})
        if r and g:
            diff_rouge = r["rouge_l_f1"] - g["rouge_l_f1"]
            diff_cos = r["cosine_sim"] - g["cosine_sim"]
            if abs(diff_rouge) < 0.01:
                md.append(f"- **{mode}**: routing vs global difference is **tiny** "
                          f"(ΔROUGE-L = {diff_rouge:+.4f}, Δcosine = {diff_cos:+.4f}). "
                          f"A1 is so accurate that filtering to the right intent "
                          f"offers marginal gains.")
            else:
                direction = "helps" if diff_rouge > 0 else "hurts"
                md.append(f"- **{mode}**: routing {direction} substantially "
                          f"(ΔROUGE-L = {diff_rouge:+.4f}, Δcosine = {diff_cos:+.4f}).")

    # Random baseline
    rand = all_results.get("random_routed", {})
    if rand:
        best_r = all_results.get(f"{served_mode}_routed", {})
        md.append(f"\n- **Random baseline** (same-intent): ROUGE-L = {rand['rouge_l_f1']:.4f} "
                  f"vs served {best_r['rouge_l_f1']:.4f}. "
                  f"{'Large gap shows retrieval matters.' if best_r['rouge_l_f1'] - rand['rouge_l_f1'] > 0.05 else 'Gap is modest — many same-intent responses are similar.'}")

    # Slices
    md.append("\n## Slices\n")
    md.append("### A1-Misclassified Queries\n")
    for label, m in all_results.items():
        sl = m.get("slices", {}).get("a1_misclassified")
        if sl:
            md.append(f"- **{label}**: n={sl['n']}, cosine={sl['cosine_sim']:.4f}, "
                      f"ROUGE-L={sl['rouge_l_f1']:.4f}, intent_cons={sl['intent_consistency']:.4f}")

    md.append("\n### Low-Confidence Queries\n")
    for label, m in all_results.items():
        sl = m.get("slices", {}).get("low_confidence")
        if sl:
            md.append(f"- **{label}**: n={sl['n']}, cosine={sl['cosine_sim']:.4f}, "
                      f"ROUGE-L={sl['rouge_l_f1']:.4f}, intent_cons={sl['intent_consistency']:.4f}")

    # DEV mode selection
    md.append("\n## Mode Selection (DEV ROUGE-L)\n")
    md.append("| Mode | DEV ROUGE-L |")
    md.append("|------|------------|")
    for mode, score in dev_scores.items():
        star = " ★" if mode == served_mode else ""
        md.append(f"| {mode}{star} | {score:.4f} |")

    md_text = "\n".join(md) + "\n"

    with open("outputs/a2_results.md", "w") as f:
        f.write(md_text)

    print("\n" + "=" * 70)
    print("DONE — outputs/a2_results.md written")
    print("=" * 70)
    print("\n" + md_text)


if __name__ == "__main__":
    main()
