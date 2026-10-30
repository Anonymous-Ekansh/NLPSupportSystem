"""Evaluate A1 + A2 on a split and save all metrics to outputs/metrics/."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from nlp.agent_a1 import IntentAnalyst
from nlp.agent_a2 import ResponseSpecialist
from nlp.pipeline import run_pipeline

METRICS_DIR = "outputs/metrics"


def evaluate(split: str = "test") -> dict:
    """Full evaluation.  Returns summary metrics dict."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    df = pd.read_csv(f"outputs/splits/{split}.csv")
    print(f"[eval] {split}: {len(df)} rows")

    a1 = IntentAnalyst()
    a2 = ResponseSpecialist()

    true_intents, pred_intents = [], []
    true_cats, pred_cats = [], []
    confidences, margins, retrieval_scores = [], [], []
    needs_clar, self_ok = 0, 0

    for i, row in df.iterrows():
        r = run_pipeline(str(row["instruction"]), a1, a2)
        true_intents.append(row["intent"])
        pred_intents.append(r["a1_output"]["intent"])
        true_cats.append(row["category"])
        pred_cats.append(r["a1_output"]["category"])
        confidences.append(r["a1_output"]["confidence"])
        margins.append(r["a1_output"]["margin"])
        retrieval_scores.append(r["a2_output"]["retrieval_score"])
        needs_clar += int(r["a1_output"]["needs_clarification"])
        self_ok += int(r["a2_output"]["self_check"]["check_passed"])
        if (len(true_intents) % 500) == 0:
            print(f"  {len(true_intents)}/{len(df)} …")

    n = len(true_intents)

    # --- Metrics ---
    metrics = {
        "split": split,
        "n_samples": n,
        "intent_accuracy": round(accuracy_score(true_intents, pred_intents), 4),
        "intent_f1_macro": round(f1_score(true_intents, pred_intents, average="macro", zero_division=0), 4),
        "intent_f1_weighted": round(f1_score(true_intents, pred_intents, average="weighted", zero_division=0), 4),
        "intent_precision_weighted": round(precision_score(true_intents, pred_intents, average="weighted", zero_division=0), 4),
        "intent_recall_weighted": round(recall_score(true_intents, pred_intents, average="weighted", zero_division=0), 4),
        "category_accuracy": round(accuracy_score(true_cats, pred_cats), 4),
        "category_f1_weighted": round(f1_score(true_cats, pred_cats, average="weighted", zero_division=0), 4),
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "avg_margin": round(float(np.mean(margins)), 4),
        "needs_clarification_pct": round(needs_clar / n * 100, 2),
        "avg_retrieval_score": round(float(np.mean(retrieval_scores)), 4),
        "self_check_pass_rate": round(self_ok / n * 100, 2),
    }

    with open(os.path.join(METRICS_DIR, f"{split}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Classification reports ---
    cr = classification_report(true_intents, pred_intents, zero_division=0)
    with open(os.path.join(METRICS_DIR, f"{split}_intent_report.txt"), "w") as f:
        f.write(cr)
    cat_cr = classification_report(true_cats, pred_cats, zero_division=0)
    with open(os.path.join(METRICS_DIR, f"{split}_category_report.txt"), "w") as f:
        f.write(cat_cr)

    # --- Confusion matrix ---
    labels = sorted(set(true_intents + pred_intents))
    cm = confusion_matrix(true_intents, pred_intents, labels=labels)
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Intent Confusion Matrix ({split})")
    fig.tight_layout()
    fig.savefig(os.path.join(METRICS_DIR, f"{split}_confusion.png"), dpi=150)
    plt.close(fig)

    print("\n=== Summary ===")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")
    return metrics


if __name__ == "__main__":
    import sys
    evaluate(sys.argv[1] if len(sys.argv) > 1 else "test")
