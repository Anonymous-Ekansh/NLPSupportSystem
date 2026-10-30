#!/usr/bin/env python3
"""Train Agent A1 model ladder.

Steps:
  1. Embedding bake-off (BGE-small vs MiniLM) — pick winner by DEV log-loss.
  2. Train L0–L3 on TRAIN, evaluate each on DEV.
  3. Pick served model on DEV log-loss (among L2, L3).
  4. Export served model weights to a1.npz (numpy-only inference).
  5. Temperature-scale on DEV (Platt scaling of logits).
  6. Choose TAU on DEV for >= 0.99 accuracy on accepted queries.
  7. Choose S_MIN on DEV+OOD to flag OOD while keeping <= 3 % in-domain flags.
  8. Save a1_config.json.

Usage:  python scripts/train_a1.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.normalize import normalize
from nlp.embed import embed, embed_split, EMBED_MODEL

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
MODEL_DIR = "outputs/models"
BAKEOFF_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax (2-D or 1-D)."""
    if logits.ndim == 1:
        e = np.exp(logits - logits.max())
        return e / e.sum()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _log_loss(y_true_idx: np.ndarray, probs: np.ndarray) -> float:
    """Log-loss for integer-labelled y and probability matrix."""
    n = len(y_true_idx)
    eps = 1e-15
    clipped = np.clip(probs, eps, 1 - eps)
    return -np.log(clipped[np.arange(n), y_true_idx]).mean()


def _ece(y_true_idx: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true_idx).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / len(y_true_idx) * abs(acc - conf)
    return ece


# ═══════════════════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════════════════

def load_split(name: str):
    df = pd.read_csv(f"outputs/splits/{name}.csv")
    texts = df["instruction"].apply(normalize).tolist()
    return df, texts


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    train_df, train_texts = load_split("train")
    dev_df, dev_texts = load_split("dev")

    classes = sorted(train_df["intent"].unique())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_train = np.array([class_to_idx[c] for c in train_df["intent"]])
    y_dev = np.array([class_to_idx[c] for c in dev_df["intent"]])

    # Intent → category map (majority vote)
    intent_to_cat = (
        train_df.groupby("intent")["category"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    print(f"  TRAIN: {len(train_texts)} rows, {n_classes} intents")
    print(f"  DEV:   {len(dev_texts)} rows")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Embedding bake-off
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("EMBEDDING BAKE-OFF")
    print("=" * 70)

    bakeoff_results = []
    best_emb_loss = float("inf")
    best_emb_model = BAKEOFF_MODELS[0]

    for model_name in BAKEOFF_MODELS:
        print(f"\n  → Embedding with {model_name} ...")
        tr_emb = embed_split("train", model=model_name)
        dv_emb = embed_split("dev", model=model_name)

        # Quick LogReg with C=1.0 to compare embedding quality
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                random_state=SEED, n_jobs=-1)
        lr.fit(tr_emb, y_train)
        dv_probs = lr.predict_proba(dv_emb)
        ll = log_loss(y_dev, dv_probs, labels=list(range(n_classes)))
        dv_acc = accuracy_score(y_dev, lr.predict(dv_emb))
        dv_f1 = f1_score(y_dev, lr.predict(dv_emb), average="macro", zero_division=0)

        bakeoff_results.append({
            "model": model_name,
            "dev_log_loss": round(ll, 6),
            "dev_accuracy": round(dv_acc, 4),
            "dev_macro_f1": round(dv_f1, 4),
        })
        print(f"    log-loss={ll:.6f}  acc={dv_acc:.4f}  F1={dv_f1:.4f}")

        if ll < best_emb_loss:
            best_emb_loss = ll
            best_emb_model = model_name

    print(f"\n  ★ Winner: {best_emb_model} (log-loss={best_emb_loss:.6f})")

    # Save bake-off table
    with open(os.path.join(MODEL_DIR, "embed_bakeoff.json"), "w") as f:
        json.dump({"winner": best_emb_model, "results": bakeoff_results}, f, indent=2)

    # Load winner embeddings
    train_emb = embed_split("train", model=best_emb_model)
    dev_emb = embed_split("dev", model=best_emb_model)

    # ═══════════════════════════════════════════════════════════════════
    # 3. Model Ladder
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("MODEL LADDER")
    print("=" * 70)

    ladder = {}

    # --- L0: majority class ---
    majority_class = int(np.bincount(y_train).argmax())
    l0_preds = np.full(len(y_dev), majority_class)
    l0_probs = np.zeros((len(y_dev), n_classes))
    l0_probs[:, majority_class] = 1.0
    l0_acc = accuracy_score(y_dev, l0_preds)
    l0_f1 = f1_score(y_dev, l0_preds, average="macro", zero_division=0)
    l0_ll = _log_loss(y_dev, l0_probs)
    ladder["L0"] = {
        "name": "majority class",
        "dev_accuracy": round(l0_acc, 4),
        "dev_macro_f1": round(l0_f1, 4),
        "dev_log_loss": round(l0_ll, 6),
        "servable": False,
    }
    print(f"\n  L0 (majority):  acc={l0_acc:.4f}  F1={l0_f1:.4f}  ll={l0_ll:.4f}")

    # --- L1: TF-IDF word+char_wb → LogReg ---
    print("\n  L1 (TF-IDF + LogReg) ...")
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2), max_features=30_000,
        sublinear_tf=True, min_df=2, analyzer="word",
    )
    tfidf_char = TfidfVectorizer(
        ngram_range=(2, 5), max_features=30_000,
        sublinear_tf=True, min_df=2, analyzer="char_wb",
    )
    from scipy.sparse import hstack
    X_train_word = tfidf_word.fit_transform(train_texts)
    X_train_char = tfidf_char.fit_transform(train_texts)
    X_train_l1 = hstack([X_train_word, X_train_char])

    X_dev_word = tfidf_word.transform(dev_texts)
    X_dev_char = tfidf_char.transform(dev_texts)
    X_dev_l1 = hstack([X_dev_word, X_dev_char])

    lr_l1 = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs",
                                random_state=SEED, n_jobs=-1)
    lr_l1.fit(X_train_l1, y_train)
    l1_probs = lr_l1.predict_proba(X_dev_l1)
    l1_preds = lr_l1.predict(X_dev_l1)
    l1_acc = accuracy_score(y_dev, l1_preds)
    l1_f1 = f1_score(y_dev, l1_preds, average="macro", zero_division=0)
    l1_ll = log_loss(y_dev, l1_probs, labels=list(range(n_classes)))
    ladder["L1"] = {
        "name": "TF-IDF word+char_wb → LogReg",
        "dev_accuracy": round(l1_acc, 4),
        "dev_macro_f1": round(l1_f1, 4),
        "dev_log_loss": round(l1_ll, 6),
        "servable": False,  # needs TfidfVectorizer at inference
    }
    # Save L1 predictions for later paired bootstrap
    l1_dev_preds = l1_preds.copy()
    print(f"    acc={l1_acc:.4f}  F1={l1_f1:.4f}  ll={l1_ll:.6f}")

    # --- L2: embedding → LogReg (tune C on DEV) ---
    print("\n  L2 (embedding → LogReg, tuning C) ...")
    best_c, best_l2_ll = 1.0, float("inf")
    for c in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        lr = LogisticRegression(max_iter=1000, C=c, solver="lbfgs",
                                random_state=SEED, n_jobs=-1)
        lr.fit(train_emb, y_train)
        p = lr.predict_proba(dev_emb)
        ll = log_loss(y_dev, p, labels=list(range(n_classes)))
        print(f"    C={c:6.2f}  ll={ll:.6f}")
        if ll < best_l2_ll:
            best_l2_ll = ll
            best_c = c

    lr_l2 = LogisticRegression(max_iter=1000, C=best_c, solver="lbfgs",
                                random_state=SEED, n_jobs=-1)
    lr_l2.fit(train_emb, y_train)
    l2_probs = lr_l2.predict_proba(dev_emb)
    l2_preds = lr_l2.predict(dev_emb)
    l2_acc = accuracy_score(y_dev, l2_preds)
    l2_f1 = f1_score(y_dev, l2_preds, average="macro", zero_division=0)
    ladder["L2"] = {
        "name": f"embedding → LogReg (C={best_c})",
        "dev_accuracy": round(l2_acc, 4),
        "dev_macro_f1": round(l2_f1, 4),
        "dev_log_loss": round(best_l2_ll, 6),
        "best_C": best_c,
        "servable": True,
    }
    print(f"  ★ L2 best: C={best_c}  acc={l2_acc:.4f}  F1={l2_f1:.4f}  ll={best_l2_ll:.6f}")

    # --- L3: similarity-weighted kNN ---
    print("\n  L3 (sim-weighted kNN, tuning k and temperature) ...")
    best_l3_ll = float("inf")
    best_k, best_temp = 5, 0.1

    for k in [3, 5, 7, 11, 15, 21]:
        for temp in [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]:
            # Cosine similarities (both L2-normed)
            sim_matrix = dev_emb @ train_emb.T  # (N_dev, N_train)

            probs_knn = np.zeros((len(y_dev), n_classes))
            for i in range(len(y_dev)):
                top_idx = np.argpartition(sim_matrix[i], -k)[-k:]
                top_sims = sim_matrix[i][top_idx]
                top_labels = y_train[top_idx]

                weights = np.exp(top_sims / temp)
                for lbl, w in zip(top_labels, weights):
                    probs_knn[i, lbl] += w

            # Normalise
            probs_knn = probs_knn / probs_knn.sum(axis=1, keepdims=True)
            ll = _log_loss(y_dev, probs_knn)

            if ll < best_l3_ll:
                best_l3_ll = ll
                best_k = k
                best_temp = temp

    print(f"    Best: k={best_k}, temp={best_temp:.3f}, ll={best_l3_ll:.6f}")

    # Recompute L3 predictions with best params
    sim_matrix = dev_emb @ train_emb.T
    l3_probs = np.zeros((len(y_dev), n_classes))
    for i in range(len(y_dev)):
        top_idx = np.argpartition(sim_matrix[i], -best_k)[-best_k:]
        top_sims = sim_matrix[i][top_idx]
        top_labels = y_train[top_idx]
        weights = np.exp(top_sims / best_temp)
        for lbl, w in zip(top_labels, weights):
            l3_probs[i, lbl] += w
    l3_probs = l3_probs / l3_probs.sum(axis=1, keepdims=True)
    l3_preds = l3_probs.argmax(axis=1)
    l3_acc = accuracy_score(y_dev, l3_preds)
    l3_f1 = f1_score(y_dev, l3_preds, average="macro", zero_division=0)
    ladder["L3"] = {
        "name": f"sim-weighted kNN (k={best_k}, T={best_temp})",
        "dev_accuracy": round(l3_acc, 4),
        "dev_macro_f1": round(l3_f1, 4),
        "dev_log_loss": round(best_l3_ll, 6),
        "best_k": best_k,
        "best_temp": best_temp,
        "servable": True,
    }
    print(f"  ★ L3: k={best_k}  T={best_temp}  acc={l3_acc:.4f}  F1={l3_f1:.4f}  ll={best_l3_ll:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Pick served model (lowest DEV log-loss among servable)
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("LADDER SUMMARY")
    print("=" * 70)

    for level, info in ladder.items():
        serv = "★" if info["servable"] else " "
        print(f"  {serv} {level}: {info['name']:45s}  "
              f"acc={info['dev_accuracy']:.4f}  F1={info['dev_macro_f1']:.4f}  "
              f"ll={info['dev_log_loss']:.6f}")

    servable = {k: v for k, v in ladder.items() if v["servable"]}
    served_level = min(servable, key=lambda k: servable[k]["dev_log_loss"])
    served_info = ladder[served_level]
    print(f"\n  → Served model: {served_level} ({served_info['name']})")

    # Gate: DEV Macro-F1 must be >= 0.97
    if served_info["dev_macro_f1"] < 0.97:
        print(f"\n❌ STOP: DEV Macro-F1 = {served_info['dev_macro_f1']:.4f} < 0.97")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # 5. Temperature scaling on DEV
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("TEMPERATURE SCALING")
    print("=" * 70)

    if served_level == "L2":
        # Get raw logits
        raw_logits = dev_emb @ lr_l2.coef_.T + lr_l2.intercept_  # (N, C)
        served_probs_uncal = l2_probs
    elif served_level == "L3":
        # For kNN, get the raw log-vote scores
        raw_logits = np.log(l3_probs + 1e-30) * best_temp  # undo normalisation
        served_probs_uncal = l3_probs

    ece_before = _ece(y_dev, served_probs_uncal)
    ll_before = _log_loss(y_dev, served_probs_uncal)
    print(f"  Before: ECE={ece_before:.4f}  log-loss={ll_before:.6f}")

    # Optimize temperature
    def temp_objective(T):
        scaled = _softmax(raw_logits / T)
        return _log_loss(y_dev, scaled)

    result = minimize_scalar(temp_objective, bounds=(0.1, 10.0), method="bounded")
    best_temperature = result.x

    scaled_probs = _softmax(raw_logits / best_temperature)
    ece_after = _ece(y_dev, scaled_probs)
    ll_after = _log_loss(y_dev, scaled_probs)
    print(f"  After:  ECE={ece_after:.4f}  log-loss={ll_after:.6f}  T={best_temperature:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. Choose TAU (confidence threshold)
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("THRESHOLD SELECTION: TAU (confidence)")
    print("=" * 70)

    dev_confidence = scaled_probs.max(axis=1)
    dev_preds = scaled_probs.argmax(axis=1)
    dev_correct = (dev_preds == y_dev)

    # Sweep TAU from low to high; pick lowest TAU giving >= 0.99 accuracy on accepted
    tau_candidates = np.sort(np.unique(dev_confidence))
    best_tau = 0.0
    tau_coverage = 1.0
    for tau in tau_candidates:
        accepted = dev_confidence >= tau
        if accepted.sum() == 0:
            continue
        acc_on_accepted = dev_correct[accepted].mean()
        cov = accepted.mean()
        if acc_on_accepted >= 0.99:
            best_tau = float(tau)
            tau_coverage = float(cov)
            break

    print(f"  TAU = {best_tau:.4f}  (coverage = {tau_coverage:.2%})")

    # ═══════════════════════════════════════════════════════════════════
    # 7. Choose S_MIN (similarity threshold for OOD)
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("THRESHOLD SELECTION: S_MIN (top1_similarity)")
    print("=" * 70)

    # Compute top1_similarity for DEV
    dev_top1_sim = (dev_emb @ train_emb.T).max(axis=1)  # (N_dev,)

    # Load OOD queries
    ood_path = "tests/ood_dev.txt"
    ood_texts = []
    with open(ood_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ood_texts.append(line)

    ood_emb = embed([normalize(t) for t in ood_texts], model=best_emb_model,
                    cache_tag="ood_dev")
    ood_top1_sim = (ood_emb @ train_emb.T).max(axis=1)

    print(f"  DEV top1_sim: min={dev_top1_sim.min():.4f}  "
          f"mean={dev_top1_sim.mean():.4f}  max={dev_top1_sim.max():.4f}")
    print(f"  OOD top1_sim: min={ood_top1_sim.min():.4f}  "
          f"mean={ood_top1_sim.mean():.4f}  max={ood_top1_sim.max():.4f}")

    # Sweep S_MIN: maximise OOD flagged while flagging <= 3% of DEV
    s_candidates = np.sort(np.concatenate([dev_top1_sim, ood_top1_sim]))[::-1]
    best_s_min = float(ood_top1_sim.max())  # default: flag nothing
    best_ood_flagged = 0

    for s in s_candidates:
        dev_flagged_pct = (dev_top1_sim < s).mean()
        ood_flagged = (ood_top1_sim < s).sum()
        if dev_flagged_pct <= 0.03:
            if ood_flagged >= best_ood_flagged:
                best_ood_flagged = ood_flagged
                best_s_min = float(s)

    dev_flagged_pct = (dev_top1_sim < best_s_min).mean()
    ood_flagged_pct = (ood_top1_sim < best_s_min).mean()
    print(f"  S_MIN = {best_s_min:.4f}")
    print(f"    DEV flagged: {dev_flagged_pct:.2%}")
    print(f"    OOD flagged: {ood_flagged_pct:.2%} ({best_ood_flagged}/{len(ood_texts)})")

    # ═══════════════════════════════════════════════════════════════════
    # 8. Export to a1.npz + a1_config.json
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("EXPORTING")
    print("=" * 70)

    export = {
        "classes": np.array(classes),
        "intent_to_cat": json.dumps(intent_to_cat),
        "train_embs": train_emb.astype(np.float32),
        "train_labels": y_train.astype(np.int32),
    }

    if served_level == "L2":
        export["W"] = lr_l2.coef_.T.astype(np.float32)   # (384, n_classes)
        export["b"] = lr_l2.intercept_.astype(np.float32)

    np.savez_compressed(os.path.join(MODEL_DIR, "a1.npz"), **export)

    config = {
        "served_level": served_level,
        "embed_model": best_emb_model,
        "temperature": round(best_temperature, 6),
        "tau": round(best_tau, 6),
        "tau_coverage": round(tau_coverage, 4),
        "s_min": round(best_s_min, 6),
        "s_min_dev_flagged_pct": round(dev_flagged_pct, 4),
        "s_min_ood_flagged_pct": round(ood_flagged_pct, 4),
        "knn_k": best_k if served_level == "L3" else None,
        "knn_temp": best_temp if served_level == "L3" else None,
        "l2_C": best_c if served_level == "L2" else None,
        "ece_before_scaling": round(ece_before, 6),
        "ece_after_scaling": round(ece_after, 6),
        "ladder": ladder,
        "embed_bakeoff": bakeoff_results,
    }

    with open(os.path.join(MODEL_DIR, "a1_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    npz_size = os.path.getsize(os.path.join(MODEL_DIR, "a1.npz")) / 1e6
    print(f"  a1.npz:         {npz_size:.1f} MB")
    print(f"  a1_config.json: saved")
    print(f"  Served level:   {served_level}")
    print(f"  Temperature:    {best_temperature:.4f}")
    print(f"  TAU:            {best_tau:.4f} (coverage {tau_coverage:.2%})")
    print(f"  S_MIN:          {best_s_min:.4f}")
    print("\n✅ A1 training complete.")


if __name__ == "__main__":
    main()
