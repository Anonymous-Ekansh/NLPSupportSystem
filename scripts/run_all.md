# Full pipeline reproduction (`scripts/run_all.py`)

Regenerates the Day-1 model artifacts, evaluation benchmarks, 2D map and frontend evidence
directly from `data/`. Level-2 evidence and submission files are produced by separate scripts
(see the last section).

```bash
source .venv/bin/activate
python scripts/run_all.py
```

---

## What `run_all.py` runs

| Step | Script | Input | Output |
|---|---|---|---|
| **1** | `scripts/train_a1.py` | `outputs/splits/train.csv` (15,567 rows) | `outputs/models/a1_pipe.joblib`, `a1.npz`, `a1_config.json`, `a1_intent_to_cat.joblib`, `a1_centroids.joblib` |
| **2** | `scripts/eval_a1.py` | `outputs/splits/test.csv` (3,475 rows), model | `outputs/evidence/a1.json`, `outputs/evidence/a1_errors.csv` |
| **3** | `scripts/eval_a2.py` | test + train splits | `outputs/evidence/a2.json`, `outputs/a2_results.md`, `outputs/a2_samples.csv` |
| **4** | `scripts/build_map.py` | `outputs/splits/train.csv` | `outputs/map_points.json` (2D projection of a stratified sample) |
| **5** | `scripts/sync_evidence.py` | `outputs/evidence/*.json`, `map_points.json` | `web/public/evidence/*`, `web/public/map_points.json` |

Runtime is a few minutes on a modern CPU; the first run also downloads the embedding models.

> `outputs/models/` and `outputs/splits/` are committed on purpose, so a fresh clone can serve
> requests without running this script.

---

## Prerequisites

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-nlp.txt
```

Splits must exist before training. If `outputs/splits/` is missing:

```bash
python -m nlp.split_data      # dedup + leak removal → train 15,567 / dev 1,730 / test 3,475
```

---

## Step details

### 1. Train A1 (`train_a1.py`)
- Embeds TRAIN instructions with `sentence-transformers/all-MiniLM-L6-v2` through fastembed
  (CPU / ONNX), 384 dimensions.
- Compares a model ladder on DEV (majority class, TF-IDF + LogReg, embedding + LogReg,
  similarity-weighted kNN) and serves the embedding + LogReg model (C chosen on DEV; 50.0 in the
  committed artifacts).
- Fits temperature scaling on DEV logits — not a CV Platt fit — and records `temperature`, `tau`
  (clarification threshold) and `s_min` (similarity floor) in `a1_config.json`.
- Writes the ONNX embedding cache matrix used for similarity and the confusion/centroid artifacts.

### 2. Evaluate A1 (`eval_a1.py`)
- Intent top-1 accuracy, top-3, macro-F1 with bootstrap CI, category accuracy, per-intent F1,
  ECE before/after calibration, OOD AUROC, and per-`flags`-letter F1.
- Writes `outputs/evidence/a1.json` and the misclassified rows to `a1_errors.csv`.

### 3. Evaluate A2 (`eval_a2.py`)
- Benchmarks `dense`, `sparse`, `hybrid` and unrouted retrieval on TEST: cosine to the reference
  reply, ROUGE-L, intent consistency, placeholder safety, self-check rejection/pass rates, plus
  slices for A1's misclassified and low-confidence rows.
- Builds A2 through `A2Builder.build()`, which loads the committed
  `outputs/models/a2_emb_matrix.joblib` / `a2_train_df.pkl` if present and otherwise recomputes
  the TRAIN embeddings (cached under `outputs/cache/`). These matrices back the in-memory
  retrieval used when Supabase is unavailable; they are written by `nlp/agent_a2.py`.
- Writes `outputs/evidence/a2.json`, `outputs/a2_results.md`, `outputs/a2_samples.csv`.

### 4. Semantic map (`build_map.py`)
- UMAP if `umap-learn` is installed, otherwise a sklearn TSNE fallback, on a stratified sample of
  TRAIN queries; coordinates normalised to `[-1, 1]` and written to `outputs/map_points.json`.

### 5. Evidence sync (`sync_evidence.py`)
- Copies evidence JSONs and the map into `web/public/` so the UI needs no backend to render
  measured numbers.

---

## Level 2 artifacts (separate commands)

`run_all.py` covers Day 1 only. To regenerate the Level-2 outputs:

| Artifact | Command |
|---|---|
| Decision thresholds (DEV only) | `python scripts/tune_a3_policy.py` |
| Robustness / stress-set evidence | `python scripts/eval_l2.py --n 500 --n-pert 100 --seed 13 --sync` |
| Cached demo payloads for the UI | `python scripts/build_demo_cache.py` |
| Submission files | `python scripts/make_submission.py --n 300 --seed 13` |
| Submission validation | `python scripts/validate_submission.py outputs/submission/predictions.csv --gold outputs/submission/gold.csv` |
| Test scripts | `python tests/test_a4.py`, `python tests/test_pipeline_l2.py` |

`eval_l2.py` writes `outputs/evidence/robustness.json`, `l2_decisions.json` and
`trace_examples.json`; the `--sync` flag copies them to `web/public/evidence/`.
