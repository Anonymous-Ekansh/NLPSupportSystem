# Full Pipeline Reproduction Guide (`scripts/run_all.py`)

This document explains how to regenerate all model artifacts, evaluation benchmarks, 2D semantic maps, and frontend evidence files directly from raw data in `data/`.

---

## Overview

The entire NLP pipeline is reproducible with a single command:

```bash
python3 scripts/run_all.py
```

### Pipeline Execution Order

| Step | Script | Input | Output Artifact | Typical Runtime |
|---|---|---|---|---|
| **1** | `scripts/train_a1.py` | `data/train.csv` | `outputs/a1_model.joblib` | ~45s |
| **2** | `scripts/eval_a1.py` | `data/test.csv`, `a1_model.joblib` | `outputs/evidence/a1.json`, `a1_errors.csv` | ~15s |
| **3** | `scripts/eval_a2.py` | `data/test.csv`, `data/train.csv` | `outputs/evidence/a2.json`, `a2_results.md`, `a2_samples.csv` | ~60s |
| **4** | `scripts/build_map.py` | `data/train.csv` | `outputs/map_points.json` (3,000 UMAP 2D points) | ~25s |
| **5** | `scripts/sync_evidence.py` | `outputs/evidence/*.json`, `map_points.json` | `web/public/evidence/*`, `web/public/map_points.json` | <1s |

**Total End-to-End Runtime**: ~2.5 to 3 minutes on CPU (Apple Silicon / modern x86).

---

## Prerequisites

Ensure all dependencies from both requirements files are installed in your virtual environment:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Install requirements
pip install -r requirements-nlp.txt
pip install -r requirements-api.txt
```

---

## Step-by-Step Breakdown

### 1. Training Agent A1 (`train_a1.py`)
- Computes 384-dimensional dense embeddings for all 21,497 training instructions using `BAAI/bge-small-en-v1.5` (CPU via ONNX runtime).
- Trains a multi-class `LogisticRegression` model with L2 regularization (`C=1.0`, `max_iter=1000`).
- Fits a 5-fold cross-validated Platt scaling calibrator to ensure confidence outputs represent true probabilities.
- Serializes model and metadata to `outputs/a1_model.joblib`.

### 2. Evaluating Agent A1 (`eval_a1.py`)
- Evaluates top-1 accuracy, top-3 accuracy, macro F1, and category accuracy on `data/test.csv` (2,688 queries).
- Calculates Expected Calibration Error (ECE) across 10 probability bins.
- Saves structured evaluation evidence to `outputs/evidence/a1.json` and misclassified queries to `outputs/evidence/a1_errors.csv`.

### 3. Evaluating Agent A2 (`eval_a2.py`)
- Benchmarks retrieval accuracy on the test set comparing unrouted baseline vs. category-routed retrieval.
- Evaluates Top-1 intent match, Top-3 recall, mean cosine similarity, and end-to-end latency.
- Validates the self-check hallucination guard.
- Saves results to `outputs/evidence/a2.json`, `outputs/a2_results.md`, and sample traces to `outputs/a2_samples.csv`.

### 4. Semantic Projection Map (`build_map.py`)
- Computes 2D UMAP projection (with t-SNE fallback) on a stratified sample of 3,000 training queries across all 27 intents.
- Normalizes coordinates to `[-1, 1]` range.
- Saves `outputs/map_points.json` for interactive rendering in the web dashboard.

### 5. Frontend Evidence Sync (`sync_evidence.py`)
- Copies all generated evidence files directly into `web/public/` so the Next.js static and dynamic views can consume them without hardcoding.
