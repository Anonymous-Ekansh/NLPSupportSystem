# Intelligent Customer Support & Automated Query Resolution

> Jaypee Agentic AI International Summit 2026 — Track 3: Agentic AI for NLP  
> **Level 1: Two-Agent Baseline**

## Architecture

```
Customer Query
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                 │
│  nlp/pipeline.py — trace-recording, map projection, latency tracking │
│                                                                      │
│  ┌───────────────────┐   full A1 JSON    ┌────────────────────────┐  │
│  │  A1 Intent Analyst │ ───────────────▶ │ A2 Response Specialist │  │
│  │  Embedding LogReg  │                  │ hybrid retrieval       │  │
│  │  model ladder      │                  │ + 3-stage self-check   │  │
│  │  temp-scaled       │                  │ pgvector → in-memory   │  │
│  └───────────────────┘                  └────────────────────────┘  │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐                    │
│  │  Verifier   │  │  Clarifier │  │  Escalator  │  ← Level 2 slots  │
│  │  (planned)  │  │  (planned) │  │  (planned)  │                    │
│  └────────────┘  └────────────┘  └─────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI Service (api/main.py)                                       │
│  /health  /meta  /resolve  /feedback                                 │
│  per-IP rate limit · pydantic validation · automatic run logging     │
└──────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Supabase (pgvector)                                                 │
│  examples — vector(384) + HNSW + FTS GIN                            │
│  runs — query traces (JSONB)                                         │
│  feedback — user ratings                                             │
└──────────────────────────────────────────────────────────────────────┘
```

### Vector Database

The system uses **Supabase pgvector** as its production vector store:

- **Table `examples`**: stores all TRAIN instruction/response pairs with 384-d embeddings (`BAAI/bge-small-en-v1.5` via fastembed ONNX, CPU)
- **HNSW index** (`vector_cosine_ops`, m=16, ef_construction=64) for fast ANN search
- **Full-text search** via `tsvector` column with GIN index
- **Two RPC functions**:
  - `match_vector(embedding, count, filter_intents)` — pure vector cosine search
  - `match_hybrid(embedding, query_text, count, filter_intents)` — Reciprocal Rank Fusion (k=60) of vector + `websearch_to_tsquery` FTS. Query text is sanitized (strips `{{…}}` and punctuation)
- **Fallback**: if Supabase is unreachable or times out (2s), the API automatically falls back to the in-memory index (exact TF-IDF + embedding hybrid with RRF). The `retrieval_backend` field in every response reports which was used.
- **RLS enabled** on all tables with no public policies (service_role key only)
- **2D map coordinates** (UMAP, stratified 3000-point sample) stored in `map_x`/`map_y` columns for visualization

### Pipeline Output

`POST /resolve` returns:
```json
{
    "query": "How can I request a refund?",
    "trace": [
        {"agent": "IntentAnalyst", "input": {...}, "output": {...}, "latency_ms": 162.2},
        {"agent": "ResponseSpecialist", "input": {"query": "...", "a1_output": {...}}, "output": {...}, "latency_ms": 1153.6}
    ],
    "final_response": "I comprehend that you need assistance with requesting a refund...",
    "total_latency_ms": 1317.5,
    "map": {"x": -3.42, "y": 1.87, "confidence": 0.84},
    "retrieval_backend": "pgvector",
    "planned_agents": [{"name": "Verifier", "status": "planned"}, ...]
}
```

## A1 Results (on TEST split)

All numbers from [`outputs/a1_results.md`](outputs/a1_results.md):

| Metric | Value |
|--------|-------|
| **Served model** | L2 (embedding → LogReg, C=50.0) |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Intent Macro-F1** | **99.52%** [95% CI: 99.27–99.74] |
| Intent Accuracy | 99.54% |
| Category Accuracy | 99.83% |
| Calibration ECE | 0.0027 (after temp scaling T=0.4828) |
| OOD AUROC (top1_sim) | 0.9962 |
| Total errors | 16 / 3475 (0.46%) |

### Model Ladder (TEST)

| Level | Name | Macro-F1 | 95% CI |
|-------|------|----------|--------|
| L0 | majority class | 0.0031 | — |
| L1 | TF-IDF word+char_wb → LogReg | **0.9988** | [0.9973, 1.000] |
| **L2 ★** | **embedding → LogReg (C=50.0)** | **0.9952** | [0.9927, 0.9974] |
| L3 | sim-weighted kNN (k=15, T=0.05) | 0.9913 | [0.9873, 0.9947] |

## Data Split

| | Rows |
|--|------|
| **Final TRAIN** | **15,567** |
| **Final DEV** | **1,730** |
| **Final TEST** | **3,475** |

- **27 intents**, **11 categories**. Seed 42 everywhere. Stratified by intent.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements-api.txt

# 2. Copy and fill .env
cp .env.example .env   # add SUPABASE_URL, SUPABASE_SERVICE_KEY

# 3. Run Supabase schema (paste supabase/schema.sql in SQL editor)

# 4. Train models + ingest data
PYTHONPATH=. python scripts/train_a1.py
PYTHONPATH=. python scripts/build_map.py
PYTHONPATH=. python scripts/ingest.py

# 5. Start API
PYTHONPATH=. uvicorn api.main:app --port 8000

# 6. Test
curl -X POST http://localhost:8000/resolve \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to cancel my subscription"}'
```

## Project Structure

```
├── data/                    # Raw CSVs (train.csv, validation.csv)
├── nlp/                     # Core NLP package
│   ├── normalize.py         # Shared text normalization
│   ├── split_data.py        # Dedup + leak removal + stratified split
│   ├── embed.py             # fastembed wrapper with caching + bake-off
│   ├── a1.py                # A1: model ladder, temp-scaled inference
│   ├── a2.py                # A2: sparse/dense/hybrid retrieval + self-check
│   └── pipeline.py          # Orchestrator with Agent interface + trace
├── api/                     # FastAPI service
│   └── main.py              # /health /meta /resolve /feedback
├── supabase/
│   └── schema.sql           # pgvector tables, HNSW, FTS, RPC functions
├── scripts/
│   ├── train_a1.py          # Full A1 training with bake-off
│   ├── eval_a1.py           # A1 evaluation (bootstrap CIs)
│   ├── eval_a2.py           # A2 evaluation (retrieval mode comparison)
│   ├── build_map.py         # UMAP/TSNE 2D projection
│   ├── ingest.py            # Supabase data ingestion
│   ├── parity_bench.py      # pgvector vs in-memory benchmark
│   ├── run_all.py           # Master runner
│   └── check_env.py         # Environment validator
├── outputs/                 # All generated artifacts (gitignored binaries)
├── tests/                   # OOD test data
├── render.yaml              # Render deployment config
├── requirements-api.txt     # API dependencies (no torch)
├── requirements-nlp.txt     # Full NLP/training dependencies
├── SETUP.md                 # Living setup checklist
└── README.md
```

## API Endpoints

Start locally: `PYTHONPATH=. uvicorn api.main:app --port 8000`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model loaded, Supabase reachable, retrieval backend |
| `GET` | `/meta` | Model names, thresholds, planned agents, headline numbers |
| `POST` | `/resolve` | Full A1→A2 pipeline (trimmed, 1-500 chars, rate-limited) |
| `POST` | `/feedback` | User feedback (query, rating 1-5, optional note) |

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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Classification | Embedding LogReg (model ladder L0–L3), temp-scaled |
| Embeddings | fastembed / sentence-transformers/all-MiniLM-L6-v2 (384-d, ONNX, CPU) |
| Retrieval | Hybrid RRF (sparse TF-IDF + dense embedding), intent-filtered |
| Vector DB | Supabase pgvector (HNSW + FTS), in-memory fallback |
| API | FastAPI + Uvicorn, pydantic, per-IP rate limit |
| Deployment | Render (Python API), Vercel (Next.js UI – planned) |
| Languages | Python 3.13 |
