#!/usr/bin/env python3
"""Evaluate Agent A1 on TEST — run ONCE, produce outputs/a1_results.md,
outputs/evidence/a1.json, and figures.

Metrics per ladder level:
  - Intent Macro-F1 with 95% bootstrap CI (1000 resamples, seed 42)
  - Accuracy, category accuracy
  - Paired-bootstrap CI of F1 difference (served vs L1)
  - Per-intent F1
  - Top-10 confusions
  - Macro-F1 per flag letter
  - ECE before & after temperature scaling + reliability diagram
  - Risk-coverage curve
  - OOD AUROC (max-prob vs top1_similarity)
  - Errors CSV

Usage:  python scripts/eval_a1.py
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.normalize import normalize
from nlp.embed import embed, embed_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
MODEL_DIR = "outputs/models"
EVIDENCE_DIR = "outputs/evidence"
FIGURES_DIR = "outputs/figures"

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        e = np.exp(logits - logits.max())
        return e / e.sum()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    c = np.clip(p, eps, 1 - eps)
    return -np.log(c[np.arange(len(y)), y]).mean()


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15):
    """Returns ECE and per-bin (acc, conf, count)."""
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins_info = []
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > boundaries[i]) & (conf <= boundaries[i + 1])
        if mask.sum() == 0:
            bins_info.append((0.0, 0.0, 0))
            continue
        a = correct[mask].mean()
        c = conf[mask].mean()
        n = int(mask.sum())
        ece += n / len(y) * abs(a - c)
        bins_info.append((float(a), float(c), n))
    return float(ece), bins_info


def bootstrap_f1(y_true, y_pred, n_boot=1000, seed=42):
    """Bootstrap 95% CI for macro-F1."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        s = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        scores.append(s)
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, n_boot=1000, seed=42):
    """Paired bootstrap 95% CI of F1(A) - F1(B)."""
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        fa = f1_score(y_true[idx], y_pred_a[idx], average="macro", zero_division=0)
        fb = f1_score(y_true[idx], y_pred_b[idx], average="macro", zero_division=0)
        diffs.append(fa - fb)
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load config
    with open(os.path.join(MODEL_DIR, "a1_config.json")) as f:
        config = json.load(f)

    served_level = config["served_level"]
    embed_model = config.get("embed_model", "BAAI/bge-small-en-v1.5")
    temperature = config["temperature"]
    tau = config["tau"]
    s_min = config["s_min"]
    ladder_info = config["ladder"]

    blob = np.load(os.path.join(MODEL_DIR, "a1.npz"), allow_pickle=True)
    classes = blob["classes"].tolist()
    intent_to_cat = json.loads(str(blob["intent_to_cat"]))
    train_embs = blob["train_embs"]
    train_labels = blob["train_labels"]

    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Load test data
    test_df = pd.read_csv("outputs/splits/test.csv")
    test_texts = test_df["instruction"].apply(normalize).tolist()
    y_test = np.array([class_to_idx[c] for c in test_df["intent"]])
    y_test_cat = test_df["category"].values

    # Load train data for L1
    train_df = pd.read_csv("outputs/splits/train.csv")
    train_texts = train_df["instruction"].apply(normalize).tolist()
    y_train = np.array([class_to_idx[c] for c in train_df["intent"]])

    print("=" * 70)
    print(f"A1 EVALUATION ON TEST ({len(test_df)} rows)")
    print("=" * 70)

    # ── Embed test ────────────────────────────────────────────────────
    test_emb = embed_split("test", model=embed_model)
    train_emb = embed_split("train", model=embed_model)

    # ═══════════════════════════════════════════════════════════════════
    # Predict with each ladder level
    # ═══════════════════════════════════════════════════════════════════

    level_preds = {}
    level_probs = {}

    # --- L0 ---
    majority = int(np.bincount(y_train).argmax())
    level_preds["L0"] = np.full(len(y_test), majority)
    p0 = np.zeros((len(y_test), n_classes))
    p0[:, majority] = 1.0
    level_probs["L0"] = p0

    # --- L1 ---
    print("\n  Computing L1 (TF-IDF + LogReg) ...")
    tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000,
                                  sublinear_tf=True, min_df=2, analyzer="word")
    tfidf_char = TfidfVectorizer(ngram_range=(2, 5), max_features=30_000,
                                  sublinear_tf=True, min_df=2, analyzer="char_wb")
    X_tr_w = tfidf_word.fit_transform(train_texts)
    X_tr_c = tfidf_char.fit_transform(train_texts)
    X_tr = hstack([X_tr_w, X_tr_c])

    X_te_w = tfidf_word.transform(test_texts)
    X_te_c = tfidf_char.transform(test_texts)
    X_te = hstack([X_te_w, X_te_c])

    lr1 = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs",
                              random_state=SEED, n_jobs=-1)
    lr1.fit(X_tr, y_train)
    level_preds["L1"] = lr1.predict(X_te)
    level_probs["L1"] = lr1.predict_proba(X_te)

    # --- L2 ---
    print("  Computing L2 (embedding → LogReg) ...")
    best_c = config.get("l2_C", 1.0)
    lr2 = LogisticRegression(max_iter=1000, C=best_c, solver="lbfgs",
                              random_state=SEED, n_jobs=-1)
    lr2.fit(train_emb, y_train)
    level_preds["L2"] = lr2.predict(test_emb)
    l2_logits = test_emb @ lr2.coef_.T + lr2.intercept_
    level_probs["L2"] = _softmax(l2_logits / temperature)

    # --- L3 ---
    print("  Computing L3 (sim-weighted kNN) ...")
    knn_k = config.get("knn_k") or ladder_info["L3"].get("best_k", 5)
    knn_temp = config.get("knn_temp") or ladder_info["L3"].get("best_temp", 0.1)
    sim_matrix = test_emb @ train_emb.T
    l3_probs = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        top_idx = np.argpartition(sim_matrix[i], -knn_k)[-knn_k:]
        top_sims = sim_matrix[i][top_idx]
        top_labels = y_train[top_idx]
        weights = np.exp(top_sims / knn_temp)
        for lbl, w in zip(top_labels, weights):
            l3_probs[i, lbl] += w
    l3_probs = l3_probs / l3_probs.sum(axis=1, keepdims=True)
    # Apply temperature
    l3_log = np.log(l3_probs + 1e-30) * knn_temp
    level_probs["L3"] = _softmax(l3_log / temperature)
    level_preds["L3"] = level_probs["L3"].argmax(axis=1)

    # Served model probs (uncalibrated and calibrated)
    served_probs_uncal = None
    if served_level == "L2":
        served_probs_uncal = _softmax(l2_logits)  # before temp scaling
    elif served_level == "L3":
        served_probs_uncal = l3_probs  # before temp scaling

    served_probs = level_probs[served_level]
    served_preds = level_preds[served_level]

    # ═══════════════════════════════════════════════════════════════════
    # Metrics per level
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("LADDER RESULTS ON TEST")
    print("=" * 70)

    level_metrics = {}
    for level in ["L0", "L1", "L2", "L3"]:
        preds = level_preds[level]
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        ci_lo, ci_hi = bootstrap_f1(y_test, preds)

        # Category accuracy
        pred_cats = np.array([intent_to_cat.get(classes[p], "?") for p in preds])
        cat_acc = accuracy_score(y_test_cat, pred_cats)

        level_metrics[level] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "macro_f1_ci_lo": round(ci_lo, 4),
            "macro_f1_ci_hi": round(ci_hi, 4),
            "category_accuracy": round(cat_acc, 4),
        }

        star = " ★" if level == served_level else ""
        print(f"  {level}{star}: acc={acc:.4f}  F1={f1:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"cat_acc={cat_acc:.4f}")

    # ── Paired bootstrap: served vs L1 ────────────────────────────────
    print("\n  Paired bootstrap: served - L1")
    diff_lo, diff_hi = paired_bootstrap_ci(y_test, served_preds, level_preds["L1"])
    includes_zero = diff_lo <= 0 <= diff_hi
    print(f"    Δ F1 95% CI: [{diff_lo:.4f}, {diff_hi:.4f}]")
    if includes_zero:
        print("    → CI INCLUDES ZERO: difference is NOT statistically significant.")
    else:
        direction = "better" if diff_lo > 0 else "worse"
        print(f"    → CI excludes zero: served model is significantly {direction} than L1.")

    # ── Per-intent F1 ─────────────────────────────────────────────────
    print("\n  Per-intent F1 (served model):")
    intent_f1s = {}
    for i, cls in enumerate(classes):
        mask = y_test == i
        if mask.sum() == 0:
            intent_f1s[cls] = 0.0
            continue
        tp = ((served_preds == i) & (y_test == i)).sum()
        fp = ((served_preds == i) & (y_test != i)).sum()
        fn = ((served_preds != i) & (y_test == i)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        intent_f1s[cls] = round(f, 4)
    for cls, f in sorted(intent_f1s.items(), key=lambda x: x[1]):
        print(f"    {cls:35s}: {f:.4f}")

    # ── Top-10 confusions ─────────────────────────────────────────────
    cm = confusion_matrix(y_test, served_preds, labels=list(range(n_classes)))
    np.fill_diagonal(cm, 0)
    flat = []
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0:
                flat.append((cm[i, j], classes[i], classes[j]))
    flat.sort(reverse=True)
    top10_conf = flat[:10]
    print("\n  Top-10 confusions:")
    for cnt, true, pred in top10_conf:
        print(f"    {true:35s} → {pred:35s}: {cnt}")

    # ── Macro-F1 per flag letter ──────────────────────────────────────
    print("\n  Macro-F1 per flag letter:")
    flag_f1 = {}
    flags = test_df["flags"].values
    all_letters = sorted(set("".join(flags)))
    for letter in all_letters:
        mask = np.array([letter in str(f) for f in flags])
        if mask.sum() == 0:
            continue
        f = f1_score(y_test[mask], served_preds[mask], average="macro", zero_division=0)
        flag_f1[letter] = round(f, 4)
        print(f"    {letter}: F1={f:.4f}  (n={mask.sum()})")

    # ═══════════════════════════════════════════════════════════════════
    # ECE + reliability diagram
    # ═══════════════════════════════════════════════════════════════════

    ece_before, bins_before = _ece(y_test, served_probs_uncal)
    ece_after, bins_after = _ece(y_test, served_probs)
    print(f"\n  ECE before scaling: {ece_before:.4f}")
    print(f"  ECE after scaling:  {ece_after:.4f}")

    # Reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, bins, title, ece_val in [
        (axes[0], bins_before, "Before temp scaling", ece_before),
        (axes[1], bins_after, "After temp scaling", ece_after),
    ]:
        accs = [b[0] for b in bins]
        confs = [b[1] for b in bins]
        counts = [b[2] for b in bins]
        n_bins = len(bins)
        bin_centers = np.linspace(1/(2*n_bins), 1 - 1/(2*n_bins), n_bins)

        ax.bar(bin_centers, accs, width=1/n_bins, alpha=0.6, label="Accuracy",
               edgecolor="black", linewidth=0.5)
        ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE = {ece_val:.4f}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "a1_reliability.png"), dpi=150)
    plt.close(fig)
    print("  → Saved reliability diagram")

    # ═══════════════════════════════════════════════════════════════════
    # Risk-coverage curve
    # ═══════════════════════════════════════════════════════════════════

    confidence = served_probs.max(axis=1)
    correct = (served_preds == y_test).astype(float)

    sorted_idx = np.argsort(confidence)[::-1]
    sorted_correct = correct[sorted_idx]
    cum_correct = np.cumsum(sorted_correct)
    coverages = np.arange(1, len(y_test) + 1) / len(y_test)
    cum_acc = cum_correct / np.arange(1, len(y_test) + 1)
    risk = 1.0 - cum_acc

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, risk, linewidth=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 − accuracy)")
    ax.set_title("A1 Risk-Coverage Curve (TEST)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(risk.max() * 1.1, 0.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "a1_risk_coverage.png"), dpi=150)
    plt.close(fig)
    print("  → Saved risk-coverage curve")

    # ═══════════════════════════════════════════════════════════════════
    # OOD AUROC
    # ═══════════════════════════════════════════════════════════════════

    ood_path = "tests/ood_dev.txt"
    ood_texts = []
    with open(ood_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ood_texts.append(line)

    ood_emb = embed([normalize(t) for t in ood_texts], model=embed_model,
                    cache_tag="ood_dev")

    # OOD labels: 0 = in-domain (test), 1 = OOD
    n_id = len(y_test)
    n_ood = len(ood_texts)

    # Max-prob score: higher → more confident → in-domain
    # For in-domain: served model confidence
    id_maxprob = confidence
    # For OOD: run through served model
    ood_maxprob = []
    if served_level == "L2":
        W = blob["W"]
        b_vec = blob["b"]
        for e in ood_emb:
            logits = e @ W + b_vec
            p = _softmax(logits / temperature)
            ood_maxprob.append(p.max())
    elif served_level == "L3":
        for e in ood_emb:
            sims = train_embs @ e
            top_idx = np.argpartition(sims, -knn_k)[-knn_k:]
            top_sims = sims[top_idx]
            top_labels = train_labels[top_idx]
            weights = np.exp(top_sims / knn_temp)
            votes = np.zeros(n_classes)
            for lbl, w in zip(top_labels, weights):
                votes[lbl] += w
            log_v = np.log(votes + 1e-30) * knn_temp
            p = _softmax(log_v / temperature)
            ood_maxprob.append(p.max())
    ood_maxprob = np.array(ood_maxprob)

    # Top1 similarity
    id_top1sim = (test_emb @ train_emb.T).max(axis=1)
    ood_top1sim = (ood_emb @ train_emb.T).max(axis=1)

    # AUROC: label 1 = in-domain, 0 = OOD → higher score = in-domain
    y_ood_labels = np.concatenate([np.ones(n_id), np.zeros(n_ood)])

    scores_maxprob = np.concatenate([id_maxprob, ood_maxprob])
    scores_top1sim = np.concatenate([id_top1sim, ood_top1sim])

    auroc_maxprob = roc_auc_score(y_ood_labels, scores_maxprob)
    auroc_top1sim = roc_auc_score(y_ood_labels, scores_top1sim)

    print(f"\n  OOD AUROC (max-prob):       {auroc_maxprob:.4f}")
    print(f"  OOD AUROC (top1_similarity): {auroc_top1sim:.4f}")
    print("  ⚠ Note: ood_dev.txt informed S_MIN, so this AUROC is optimistic.")

    # ═══════════════════════════════════════════════════════════════════
    # Errors CSV
    # ═══════════════════════════════════════════════════════════════════

    errors_mask = served_preds != y_test
    errors_df = test_df[errors_mask].copy()
    errors_df["predicted_intent"] = [classes[p] for p in served_preds[errors_mask]]
    errors_df["predicted_category"] = [
        intent_to_cat.get(classes[p], "?") for p in served_preds[errors_mask]
    ]
    errors_df["confidence"] = confidence[errors_mask]
    errors_df["top1_similarity"] = id_top1sim[errors_mask]
    errors_df.to_csv(os.path.join(EVIDENCE_DIR, "a1_errors.csv"), index=False)
    print(f"\n  Errors: {errors_mask.sum()}/{len(y_test)} → saved a1_errors.csv")

    # ═══════════════════════════════════════════════════════════════════
    # Confusion matrix figure
    # ═══════════════════════════════════════════════════════════════════

    full_cm = confusion_matrix(y_test, served_preds, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(full_cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"A1 Intent Confusion Matrix (TEST, {served_level})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "a1_confusion.png"), dpi=150)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════
    # Build evidence JSON
    # ═══════════════════════════════════════════════════════════════════

    evidence = {
        "served_level": served_level,
        "embed_model": embed_model,
        "temperature": temperature,
        "tau": tau,
        "s_min": s_min,
        "n_test": len(y_test),
        "level_metrics": level_metrics,
        "paired_bootstrap_served_vs_l1": {
            "ci_lo": round(diff_lo, 4),
            "ci_hi": round(diff_hi, 4),
            "includes_zero": includes_zero,
        },
        "per_intent_f1": intent_f1s,
        "top10_confusions": [
            {"true": t, "pred": p, "count": int(c)} for c, t, p in top10_conf
        ],
        "flag_f1": flag_f1,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "ood_auroc_maxprob": round(auroc_maxprob, 4),
        "ood_auroc_top1sim": round(auroc_top1sim, 4),
        "n_errors": int(errors_mask.sum()),
    }

    with open(os.path.join(EVIDENCE_DIR, "a1.json"), "w") as f:
        json.dump(evidence, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    # Build a1_results.md
    # ═══════════════════════════════════════════════════════════════════

    md = []
    md.append("# A1 Intent Analyst — Evaluation Results\n")
    md.append(f"**Served model:** {served_level} ({ladder_info[served_level]['name']})\n")
    md.append(f"**Embedding model:** `{embed_model}`\n")
    md.append(f"**Temperature:** {temperature:.4f}  |  "
              f"**TAU:** {tau:.4f}  |  **S_MIN:** {s_min:.4f}\n")
    md.append(f"**Test set:** {len(y_test)} queries\n")

    # Embedding bake-off
    md.append("\n## Embedding Bake-off\n")
    md.append("| Model | DEV log-loss | DEV accuracy | DEV Macro-F1 |")
    md.append("|-------|-------------|-------------|-------------|")
    for r in config["embed_bakeoff"]:
        star = " ★" if r["model"] == embed_model else ""
        md.append(f"| `{r['model']}`{star} | {r['dev_log_loss']:.6f} | "
                  f"{r['dev_accuracy']:.4f} | {r['dev_macro_f1']:.4f} |")

    # Ladder
    md.append("\n## Model Ladder (TEST)\n")
    md.append("| Level | Name | Accuracy | Macro-F1 | 95% CI | Cat Acc |")
    md.append("|-------|------|----------|----------|--------|---------|")
    for level in ["L0", "L1", "L2", "L3"]:
        m = level_metrics[level]
        info = ladder_info[level]
        star = " ★" if level == served_level else ""
        md.append(f"| {level}{star} | {info['name']} | {m['accuracy']:.4f} | "
                  f"{m['macro_f1']:.4f} | [{m['macro_f1_ci_lo']:.4f}, {m['macro_f1_ci_hi']:.4f}] | "
                  f"{m['category_accuracy']:.4f} |")

    # Paired bootstrap
    md.append(f"\n**Paired bootstrap ({served_level} − L1):** "
              f"ΔF1 95% CI = [{diff_lo:.4f}, {diff_hi:.4f}]")
    if includes_zero:
        md.append("→ CI **includes zero**: the difference is **not statistically significant**.\n")
    else:
        direction = "better" if diff_lo > 0 else "worse"
        md.append(f"→ CI **excludes zero**: the served model is significantly "
                  f"**{direction}** than L1.\n")

    # Per-intent F1
    md.append("\n## Per-Intent F1 (served model)\n")
    md.append("| Intent | F1 |")
    md.append("|--------|------|")
    for cls in classes:
        md.append(f"| {cls} | {intent_f1s[cls]:.4f} |")

    # Top-10 confusions
    md.append("\n## Top-10 Confusions\n")
    md.append("| True | Predicted | Count |")
    md.append("|------|-----------|-------|")
    for cnt, true, pred in top10_conf:
        md.append(f"| {true} | {pred} | {cnt} |")

    # Flag F1
    md.append("\n## Macro-F1 per Flag Letter\n")
    md.append("| Flag | Macro-F1 | n |")
    md.append("|------|----------|---|")
    for letter in sorted(flag_f1.keys()):
        n_flag = sum(1 for f in flags if letter in str(f))
        md.append(f"| {letter} | {flag_f1[letter]:.4f} | {n_flag} |")

    # Calibration
    md.append("\n## Calibration\n")
    md.append(f"| | ECE |")
    md.append(f"|--|-----|")
    md.append(f"| Before temperature scaling | {ece_before:.4f} |")
    md.append(f"| After temperature scaling (T={temperature:.4f}) | {ece_after:.4f} |")
    md.append(f"\n![Reliability Diagram](outputs/figures/a1_reliability.png)\n")

    # Risk-coverage
    md.append("\n## Risk-Coverage\n")
    md.append(f"![Risk-Coverage Curve](outputs/figures/a1_risk_coverage.png)\n")

    # OOD
    md.append("\n## OOD Detection\n")
    md.append("| Detector | AUROC |")
    md.append("|----------|-------|")
    md.append(f"| max-prob | {auroc_maxprob:.4f} |")
    md.append(f"| top1_similarity | {auroc_top1sim:.4f} |")
    md.append(f"\n> ⚠ `ood_dev.txt` informed `S_MIN`, so this AUROC is **optimistic**.\n")

    # Thresholds
    md.append("\n## Thresholds\n")
    md.append(f"- **TAU** = {tau:.4f} (coverage = {config['tau_coverage']:.2%})")
    md.append(f"- **S_MIN** = {s_min:.4f} "
              f"(DEV flagged = {config['s_min_dev_flagged_pct']:.2%}, "
              f"OOD flagged = {config['s_min_ood_flagged_pct']:.2%})")

    # Errors
    md.append(f"\n## Errors\n")
    md.append(f"Total errors: **{errors_mask.sum()}** / {len(y_test)} "
              f"({errors_mask.mean():.2%})")
    md.append(f"\nSee `outputs/evidence/a1_errors.csv` for full error list.\n")

    md_text = "\n".join(md)
    with open("outputs/a1_results.md", "w") as f:
        f.write(md_text)

    print("\n" + "=" * 70)
    print("DONE — outputs/a1_results.md written")
    print("=" * 70)

    # Print it
    print("\n" + md_text)


if __name__ == "__main__":
    main()
