#!/usr/bin/env python3
"""Build the JSON files consumed by the ResolveIQ web app.

Everything here is derived from artefacts already produced by the NLP
pipeline (splits, A1/A2 evidence, cached embeddings). Nothing is hand-typed,
so the UI can keep its promise that every number is measured.

Usage:  .venv/bin/python scripts/build_web_data.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.a2 import _extract_placeholders  # noqa: E402
from nlp.embed import embed  # noqa: E402
from nlp.normalize import normalize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "public", "data")
MODELS = os.path.join(ROOT, "outputs", "models")
CACHE = os.path.join(ROOT, "outputs", "cache")
SPLITS = os.path.join(ROOT, "outputs", "splits")
EVIDENCE = os.path.join(ROOT, "outputs", "evidence")

SEED = 42
os.makedirs(OUT, exist_ok=True)


def read_json(path: str):
    with open(path) as handle:
        return json.load(handle)


def write(name: str, payload) -> None:
    path = os.path.join(OUT, name)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def pct(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = SEED):
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = values[rng.choice(n, n, replace=True)].mean()
    return [pct(np.percentile(means, 2.5)), pct(np.percentile(means, 97.5))]


# ═══════════════════════════════════════════════════════════════════════
# Load shared inputs
# ═══════════════════════════════════════════════════════════════════════

print("Loading pipeline artefacts …")
config = read_json(os.path.join(MODELS, "a1_config.json"))
a1_evidence = read_json(os.path.join(EVIDENCE, "a1.json"))
a2_evidence = read_json(os.path.join(EVIDENCE, "a2.json"))
test_metrics = read_json(os.path.join(ROOT, "outputs", "metrics", "test_metrics.json"))
split_report = read_json(os.path.join(SPLITS, "split_report.json"))

train_df = pd.read_csv(os.path.join(SPLITS, "train.csv"))
dev_df = pd.read_csv(os.path.join(SPLITS, "dev.csv"))
test_df = pd.read_csv(os.path.join(SPLITS, "test.csv"))
full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

blob = np.load(os.path.join(MODELS, "a1.npz"), allow_pickle=True)
classes: list[str] = blob["classes"].tolist()
intent_to_cat: dict[str, str] = json.loads(str(blob["intent_to_cat"]))
train_emb = blob["train_embs"].astype(np.float64)
W = blob["W"].astype(np.float64)
b = blob["b"].astype(np.float64)
temperature = float(config["temperature"])
tau = float(config["tau"])
s_min = float(config["s_min"])

test_emb = np.load(
    os.path.join(
        CACHE,
        "test_sentence-transformers__all-MiniLM-L6-v2_ddf8f9379d8a81ef.npy",
    )
).astype(np.float64)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════
# A1 predictions on TEST
# ═══════════════════════════════════════════════════════════════════════

print("Re-running A1 predictions (cached embeddings) …")
raw_probs = softmax((test_emb @ W + b) / temperature)
y_pred_idx = raw_probs.argmax(axis=1)
y_pred = [classes[i] for i in y_pred_idx]
y_true = test_df["intent"].astype(str).tolist()
label_to_idx = {name: index for index, name in enumerate(classes)}
y_true_idx = np.array([label_to_idx[name] for name in y_true])
cat_pred = [intent_to_cat.get(intent, "UNKNOWN") for intent in y_pred]
cat_true = test_df["category"].astype(str).tolist()

confidence = raw_probs.max(axis=1)
# Top-1 nearest TRAIN neighbour, exactly as A1 computes it.
top1_similarity = (test_emb @ train_emb.T).max(axis=1)
needs_clarification = (confidence < tau) | (top1_similarity < s_min)

print(f"  intent accuracy  {np.mean(np.array(y_pred) == np.array(y_true)):.4f}")
print(f"  category accuracy {np.mean(np.array(cat_pred) == np.array(cat_true)):.4f}")

# ═══════════════════════════════════════════════════════════════════════
# Performance metrics
# ═══════════════════════════════════════════════════════════════════════

from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

intent_labels = sorted(set(y_true) | set(y_pred))
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=intent_labels, zero_division=0
)

per_intent = {
    "metrics": [
        {
            "intent": intent,
            "category": intent_to_cat.get(intent, "UNKNOWN"),
            "f1": pct(f1[i]),
            "precision": pct(precision[i]),
            "recall": pct(recall[i]),
            "support": int(support[i]),
        }
        for i, intent in enumerate(intent_labels)
    ]
}

cat_labels = sorted(set(cat_true) | set(cat_pred))
cat_matrix = confusion_matrix(cat_true, cat_pred, labels=cat_labels)
cat_confusion = {
    "labels": cat_labels,
    "matrix": cat_matrix.astype(int).tolist(),
    "note": "Rows are the true category; columns are the predicted category.",
}

# ═══════════════════════════════════════════════════════════════════════
# Reliability bins (before / after temperature scaling)
# ═══════════════════════════════════════════════════════════════════════

def reliability(probs: np.ndarray, bins: int = 12):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (y_true_idx == pred).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        n = int(mask.sum())
        ece += n / len(conf) * abs(acc - avg_conf)
        rows.append(
            {
                "confidence": pct(avg_conf),
                "accuracy": pct(acc),
                "count": n,
            }
        )
    return rows, float(ece)


unscaled = softmax(test_emb @ W + b)
scaled = softmax((test_emb @ W + b) / temperature)
rel_before, ece_before = reliability(unscaled)
rel_after, ece_after = reliability(scaled)

# ═══════════════════════════════════════════════════════════════════════
# A2 served-mode replay: answer relevance + per-check outcomes
# ═══════════════════════════════════════════════════════════════════════

print("Replaying A2 dense-routed mode for answer checks …")
a2 = joblib.load(os.path.join(MODELS, "a2_specialist.joblib"))
a2_test_emb = embed(
    [normalize(text) for text in test_df["instruction"].astype(str)],
    model=a2.embed_model,
    cache_tag="web_a2_test",
).astype(np.float64)
a2_train_emb = np.asarray(a2.train_emb, dtype=np.float64)
a2_intents = np.asarray(a2.intents)
a2_responses = np.asarray(a2.responses, dtype=object)
a2_queries = a2.train_df["instruction"].astype(str).to_numpy()

placeholder_safe_arr = np.array(["{{" not in str(r) for r in a2_responses])
resp_pred = a2.resp_clf.predict(
    a2.resp_tfidf_vec.transform([str(r) for r in a2_responses])
)
resp_placeholders = [_extract_placeholders(str(r)) for r in a2_responses]
query_placeholders = [_extract_placeholders(q) for q in test_df["instruction"].astype(str)]

ret_emb = np.load(
    os.path.join(
        CACHE,
        "a2_ret_dense_routed_sentence-transformers__all-MiniLM-L6-v2_ca160032999ccbd0.npy",
    )
).astype(np.float64)
ref_emb = np.load(
    os.path.join(
        CACHE,
        "a2_ref_sentence-transformers__all-MiniLM-L6-v2_7ae26abc8e744c2e.npy",
    )
).astype(np.float64)
answer_cosine = (ret_emb * ref_emb).sum(axis=1)

pass_flags: list[bool] = []
ph_flags: list[bool] = []
src_flags: list[bool] = []
resp_flags: list[bool] = []
rejected_any: list[bool] = []

for i in range(len(test_df)):
    probs = raw_probs[i]
    ranked = np.argsort(probs)[::-1]
    intent = classes[int(ranked[0])]
    clar = bool(needs_clarification[i])
    top3 = [(classes[int(ranked[k])], float(probs[int(ranked[k])])) for k in range(3)]

    if not clar:
        filter_intents = {intent}
        intent_probs = {intent: 1.0}
    else:
        filter_intents = {name for name, _ in top3}
        intent_probs = {name: prob for name, prob in top3}

    indices = np.where(np.isin(a2_intents, list(filter_intents)))[0]
    if len(indices) < 3:
        indices = np.arange(len(a2_intents))
        filter_intents = set(a2_intents.tolist())

    scores = a2_train_emb[indices] @ a2_test_emb[i]
    if clar and len(filter_intents) > 1:
        for j, candidate_intent in enumerate(a2_intents[indices]):
            scores[j] *= intent_probs.get(str(candidate_intent), 0.01)

    n_cand = min(15, len(scores))
    top_local = np.argsort(scores)[::-1][:n_cand]
    top_global = indices[top_local]
    top_scores = scores[top_local]

    order = sorted(
        range(len(top_global)),
        key=lambda j: (
            0
            if resp_placeholders[int(top_global[j])].issubset(query_placeholders[i])
            else 1,
            -float(top_scores[j]),
        ),
    )
    top_global = top_global[order]

    chosen = None
    rejected = 0
    first = None
    for j in range(min(5, len(top_global))):
        g = int(top_global[j])
        ph_ok = bool(placeholder_safe_arr[g])
        src_sim = float(a2_test_emb[i] @ a2_train_emb[g])
        src_ok = src_sim >= a2.sim_min
        resp_ok = str(resp_pred[g]) == intent
        passed = ph_ok and src_ok and resp_ok
        if j == 0:
            first = (ph_ok, src_ok, resp_ok, passed)
        if passed:
            chosen = (ph_ok, src_ok, resp_ok, passed)
            break
        rejected += 1

    outcome = chosen or first or (False, False, False, False)
    ph_flags.append(outcome[0])
    src_flags.append(outcome[1])
    resp_flags.append(outcome[2])
    pass_flags.append(outcome[3])
    rejected_any.append(rejected > 0)

pass_flags_arr = np.array(pass_flags, dtype=float)
ph_arr = np.array(ph_flags, dtype=float)
src_arr = np.array(src_flags, dtype=float)
resp_arr = np.array(resp_flags, dtype=float)

replay_pass_rate = float(pass_flags_arr.mean())
print(f"  replay self-check pass {replay_pass_rate:.4f} (evidence {a2_evidence['test_results']['dense_routed']['check_passed_rate']:.4f})")
print(f"  replay cosine           {answer_cosine.mean():.4f} (evidence {a2_evidence['test_results']['dense_routed']['cosine_sim']:.4f})")

answer_checks = {
    "pass_rate": pct(pass_flags_arr.mean()),
    "pass_rate_ci": bootstrap_ci(pass_flags_arr),
    "checks": [
        {
            "key": "placeholder_safe",
            "label": "Placeholders resolved",
            "pass_rate": pct(ph_arr.mean()),
            "fail_rate": pct(1 - ph_arr.mean()),
        },
        {
            "key": "source_similarity_ok",
            "label": "Source close enough",
            "pass_rate": pct(src_arr.mean()),
            "fail_rate": pct(1 - src_arr.mean()),
        },
        {
            "key": "response_intent_ok",
            "label": "Answer matches intent",
            "pass_rate": pct(resp_arr.mean()),
            "fail_rate": pct(1 - resp_arr.mean()),
        },
    ],
    "rejected_share": pct(np.array(rejected_any, dtype=float).mean()),
}

# ═══════════════════════════════════════════════════════════════════════
# Style flags → the six reader-facing style buckets
# ═══════════════════════════════════════════════════════════════════════

# Bitext flag letters overlap. Assign one primary bucket per example so the
# donut sums to 100%. Priority: the most specific reader-facing style wins.
STYLE_PRIORITY = [
    ("Polite", {"P"}),
    ("Colloquial", {"C", "W"}),
    ("Morphological", {"M"}),
    ("Noisy / misspelled", {"Z", "K", "S", "N"}),
    ("Interrogative", {"I"}),
]


def primary_style(flags: str) -> str:
    letters = set(str(flags))
    for style, markers in STYLE_PRIORITY:
        if letters & markers:
            return style
    return "Other"


full_df["style"] = full_df["flags"].map(primary_style)
style_counts = Counter(full_df["style"])
styles = [
    {
        "style": style,
        "count": int(style_counts.get(style, 0)),
        "pct": pct(style_counts.get(style, 0) / len(full_df)),
    }
    for style in ["Colloquial", "Interrogative", "Noisy / misspelled", "Morphological", "Polite", "Other"]
]

# Macro-F1 within each style bucket on TEST.
test_styles = test_df["flags"].map(primary_style)
robustness = []
for style in ["Colloquial", "Interrogative", "Noisy / misspelled", "Morphological", "Polite", "Other"]:
    mask = (test_styles == style).to_numpy()
    if mask.sum() < 5:
        continue
    robustness.append(
        {
            "style": style,
            "macro_f1": pct(
                f1_score(
                    np.array(y_true)[mask],
                    np.array(y_pred)[mask],
                    average="macro",
                    zero_division=0,
                )
            ),
            "support": int(mask.sum()),
        }
    )

# ═══════════════════════════════════════════════════════════════════════
# Dataset stats
# ═══════════════════════════════════════════════════════════════════════

category_counts = (
    full_df.groupby("category").size().sort_values(ascending=False)
)
class_counts = (
    full_df.groupby(["category", "intent"]).size().reset_index(name="count")
)
min_count = int(class_counts["count"].min())
max_count = int(class_counts["count"].max())

dataset_stats = {
    "total_examples": int(len(full_df)),
    "intents": int(full_df["intent"].nunique()),
    "categories": int(full_df["category"].nunique()),
    "splits": {
        "train": int(split_report["final_train"]),
        "dev": int(split_report["final_dev"]),
        "test": int(split_report["final_test"]),
    },
    "cleaning": {
        "raw_train_rows": int(split_report["raw_train_rows"]),
        "raw_validation_rows": int(split_report["raw_validation_rows"]),
        "duplicates_removed": int(split_report["train_duplicates_removed"]),
        "leaked_rows_removed": int(split_report["validation_leaks_removed"]),
        "seed": 42,
    },
    "category_counts": [
        {"category": str(cat), "count": int(count)}
        for cat, count in category_counts.items()
    ],
    "class_balance": [
        {
            "intent": str(row.intent),
            "category": str(row.category),
            "count": int(row.count),
        }
        for row in class_counts.sort_values("count", ascending=False).itertuples()
    ],
    "class_balance_ratio": {
        "min": min_count,
        "max": max_count,
        "ratio": pct(max_count / min_count, 2),
    },
    "styles": styles,
    "style_note": "One primary style per example; the raw flag letters can overlap.",
}

# ═══════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════

l2 = a1_evidence["level_metrics"]["L2"]
a2_dense = a2_evidence["test_results"]["dense_routed"]

results = {
    "n_test": int(a1_evidence["n_test"]),
    "intent_macro_f1": pct(l2["macro_f1"]),
    "intent_macro_f1_ci": [
        pct(l2["macro_f1_ci_lo"]),
        pct(l2["macro_f1_ci_hi"]),
    ],
    "intent_accuracy": pct(l2["accuracy"]),
    "category_accuracy": pct(l2["category_accuracy"]),
    "answer_relevance": {
        "cosine_sim": pct(answer_cosine.mean()),
        "cosine_sim_ci": bootstrap_ci(answer_cosine),
        "rouge_l_f1": pct(a2_dense["rouge_l_f1"]),
        "self_check_pass_rate": pct(pass_flags_arr.mean()),
        "self_check_pass_rate_ci": bootstrap_ci(pass_flags_arr),
        "served_mode": "dense",
    },
    "calibration": {
        "tau": pct(tau, 6),
        "s_min": pct(s_min, 6),
        "ece_before": pct(a1_evidence["ece_before"]),
        "ece_after": pct(a1_evidence["ece_after"]),
        "bins_before": rel_before,
        "bins_after": rel_after,
    },
    "answer_checks": answer_checks,
    "robustness": robustness,
    "confidence_rates": {
        "avg_confidence": pct(confidence.mean()),
        "needs_clarification": pct(needs_clarification.mean()),
        "a1_low_conf": pct(a2_evidence["a1_low_conf_rate"]),
        "category_accuracy": pct(l2["category_accuracy"]),
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Model ladder, retrieval modes, OOD
# ═══════════════════════════════════════════════════════════════════════

ladder_names = {
    "L0": "Majority class",
    "L1": "TF-IDF word + char",
    "L2": "Embedding LogReg",
    "L3": "kNN check",
}
ladder = []
for level in ["L0", "L1", "L2", "L3"]:
    metrics = a1_evidence["level_metrics"][level]
    ladder.append(
        {
            "level": level,
            "name": ladder_names[level],
            "macro_f1": pct(metrics["macro_f1"]),
            "ci_lo": pct(metrics.get("macro_f1_ci_lo", metrics["macro_f1"])),
            "ci_hi": pct(metrics.get("macro_f1_ci_hi", metrics["macro_f1"])),
            "accuracy": pct(metrics["accuracy"]),
            "served": level == config["served_level"],
        }
    )

mode_names = {"sparse": "Sparse", "dense": "Dense", "hybrid": "Hybrid"}
modes = []
for key in ["sparse", "dense", "hybrid"]:
    row = a2_evidence["test_results"][f"{key}_routed"]
    modes.append(
        {
            "key": key,
            "mode": mode_names[key],
            "cosine_sim": pct(row["cosine_sim"]),
            "rouge_l_f1": pct(row["rouge_l_f1"]),
            "intent_consistency": pct(row["intent_consistency"]),
            "check_passed_rate": pct(row["check_passed_rate"]),
            "served": key == a2_evidence["served_mode"],
        }
    )
routing = {
    "with_a1": [],
    "without_a1": [],
}
for key in ["sparse", "dense", "hybrid"]:
    routed = a2_evidence["test_results"][f"{key}_routed"]
    global_row = a2_evidence["test_results"][f"{key}_global"]
    routing["with_a1"].append(
        {
            "mode": mode_names[key],
            "key": key,
            "cosine_sim": pct(routed["cosine_sim"]),
            "intent_consistency": pct(routed["intent_consistency"]),
        }
    )
    routing["without_a1"].append(
        {
            "mode": mode_names[key],
            "key": key,
            "cosine_sim": pct(global_row["cosine_sim"]),
            "intent_consistency": pct(global_row["intent_consistency"]),
        }
    )

retrieval_modes = {
    "modes": modes,
    "routing": routing,
    "served_mode": a2_evidence["served_mode"],
    "note": "Answer relevance is cosine similarity against the held-out reference reply.",
}

with open(os.path.join(ROOT, "tests", "ood_dev.txt")) as handle:
    ood_lines = [
        line.strip()
        for line in handle
        if line.strip() and not line.strip().startswith("#")
    ]

ood = {
    "auroc_maxprob": pct(a1_evidence["ood_auroc_maxprob"]),
    "auroc_top1sim": pct(a1_evidence["ood_auroc_top1sim"]),
    "flagged_pct_in_domain": pct(config["s_min_dev_flagged_pct"]),
    "flagged_pct_ood": pct(config["s_min_ood_flagged_pct"]),
    "threshold_s_min": pct(config["s_min"], 6),
    "n_ood": len(ood_lines),
}

# ═══════════════════════════════════════════════════════════════════════
# Examples: 5 real traces + a searchable browse sample
# ═══════════════════════════════════════════════════════════════════════

traces = read_json(os.path.join(ROOT, "outputs", "examples", "traces.json"))

rng = np.random.RandomState(SEED)
browse_source = full_df.sample(n=min(160, len(full_df)), random_state=SEED)
browse = [
    {
        "query": str(row.instruction),
        "category": str(row.category),
        "intent": str(row.intent),
        "style": primary_style(row.flags),
    }
    for row in browse_source.itertuples()
]

examples = {"traces": traces, "browse": browse}

# ═══════════════════════════════════════════════════════════════════════
# Map points with the original question attached
# ═══════════════════════════════════════════════════════════════════════

map_path = os.path.join(OUT, "map_points.json")
map_points = read_json(map_path)
for point in map_points["coordinates"]:
    index = int(point["index"])
    if 0 <= index < len(train_df):
        point["query"] = str(train_df.iloc[index]["instruction"])
    else:
        point["query"] = ""

# ═══════════════════════════════════════════════════════════════════════
# Write everything
# ═══════════════════════════════════════════════════════════════════════

print("Writing web data …")
write("dataset_stats.json", dataset_stats)
write("results.json", results)
write("per_intent_f1.json", per_intent)
write("confusion_category.json", cat_confusion)
write("ladder.json", {"rows": ladder})
write("retrieval_modes.json", retrieval_modes)
write("ood.json", ood)
write("examples.json", examples)
write("map_points.json", map_points)
print("Done.")
