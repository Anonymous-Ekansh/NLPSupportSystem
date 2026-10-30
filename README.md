# Intelligent Customer Support & Automated Query Resolution

> Jaypee Agentic AI International Summit 2026 — Track 3: Agentic AI for NLP  
> **Level 1: Two-Agent Baseline**

## Architecture

```
Customer Query
      │
      ▼
┌─────────────────────┐    handoff (JSON)    ┌─────────────────────────┐
│  A1 Intent Analyst   │ ──────────────────▶ │  A2 Response Specialist  │
│  TF-IDF + LogReg     │                     │  fastembed cosine retr.  │
│  + BGE centroids     │                     │  + self-check            │
└─────────────────────┘                     └─────────────────────────┘
      │                                              │
      ▼                                              ▼
 intent, category,                           response, supporting
 confidence, margin,                         examples, retrieval
 top1_similarity,                            score, self_check
 needs_clarification
```

**A1 → A2 handoff:** A1's full output dict is passed to A2, which uses the predicted intent to filter the retrieval corpus before cosine ranking.

## Results (on TEST split)

All numbers from [`outputs/metrics/test_metrics.json`](outputs/metrics/test_metrics.json):

| Metric | Value |
|--------|-------|
| Intent Accuracy | 99.34% |
| **Intent Macro-F1** | **99.17%** |
| Intent Weighted-F1 | 99.34% |
| **Category Accuracy** | **99.83%** |
| Avg Confidence | 94.81% |
| Avg Retrieval Score | 0.9703 |
| Self-check Pass Rate | 59.48% |
| Needs Clarification | 0.43% |

> Self-check pass rate reflects `placeholder_safe` flagging template variables (`{{Account Type}}`) in training responses — expected behavior, not errors.

## Data Split

From [`outputs/splits/split_report.json`](outputs/splits/split_report.json):

| | Rows |
|--|------|
| Raw train | 18,810 |
| Raw validation | 4,031 |
| Train duplicates removed | 1,513 |
| Validation leaks removed | 528 |
| Validation duplicates removed | 28 |
| **Final TRAIN** | **15,567** |
| **Final DEV** | **1,730** |
| **Final TEST** | **3,475** |

- **27 intents**, **11 categories** (task statement says 10; data has 11).
- Seed 42 everywhere. Stratified by intent.

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn joblib matplotlib fastembed fastapi uvicorn pydantic

# 2. Verify environment
python scripts/check_env.py

# 3. Run full pipeline (split → train → evaluate → examples)
python scripts/run_all.py
```

Outputs land in `outputs/{splits,models,metrics,examples}/`.

## Project Structure

```
├── data/                    # Raw CSVs (train.csv, validation.csv)
├── nlp/                     # Core NLP package
│   ├── normalize.py         # Shared text normalization
│   ├── split_data.py        # Dedup + leak removal + stratified split
│   ├── embeddings.py        # fastembed BGE-small-en-v1.5 wrapper
│   ├── agent_a1.py          # A1: TF-IDF + LogReg intent classifier
│   ├── agent_a2.py          # A2: Embedding retrieval + self-check
│   ├── pipeline.py          # A1 → A2 chain
│   ├── evaluate.py          # Metrics, reports, confusion matrix
│   └── examples.py          # Generate end-to-end traces
├── scripts/
│   ├── run_all.py           # Master runner
│   └── check_env.py         # Environment validator
├── outputs/                 # All generated artifacts
│   ├── metrics/             # test_metrics.json, classification reports, confusion matrix
│   └── examples/            # traces.json, traces.md
├── SETUP.md                 # Living setup checklist
└── README.md
```

## Agent Contracts

**A1.analyze(query):**
```json
{
  "query": "...",
  "intent": "cancel_order",
  "category": "ORDER",
  "confidence": 0.9876,
  "margin": 0.9654,
  "top3": [{"intent": "cancel_order", "prob": 0.9876}, ...],
  "top1_similarity": 0.8234,
  "needs_clarification": false
}
```

**A2.respond(query, a1_output):**
```json
{
  "response": "...",
  "supporting_examples": [{"instruction": "...", "response": "...", "similarity": 0.95}],
  "used_intents": ["cancel_order"],
  "retrieval_score": 0.9512,
  "low_confidence": false,
  "self_check": {
    "placeholder_safe": true,
    "source_similarity_ok": true,
    "response_intent_ok": true,
    "check_passed": true,
    "candidates_rejected": 0
  }
}
```

## Planned Improvements (Level 2+)

- **A3 Clarification Agent**: triggered when `needs_clarification=true`, asks follow-up questions
- **A4 Escalation Agent**: routes to human when self-check fails or confidence is critically low
- **Response generation**: move from retrieval-only to retrieval-augmented generation
- **Supabase pgvector**: move embedding index to Postgres for persistence and scalability
- **FastAPI service + Next.js UI**: full-stack deployment on Render + Vercel

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Classification | TF-IDF (1,2)-grams + Logistic Regression |
| Embeddings | fastembed / BAAI/bge-small-en-v1.5 (384-d, ONNX, CPU) |
| Retrieval | Cosine similarity over intent-filtered corpus |
| Languages | Python 3.13 |
| Key libraries | pandas, numpy, scikit-learn, joblib, fastembed, matplotlib |
