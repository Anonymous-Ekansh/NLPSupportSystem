# Setup — ResolveIQ (Level 2)

Everything below is for the five-agent Level 2 system. Model artifacts, splits and evidence are
**committed**, so a fresh clone can serve requests without retraining.

---

## 1. Prerequisites

| Requirement | Version used here | Check |
|---|---|---|
| Python | 3.13 (3.11+ required by the pinned deps) | `python3 --version` |
| Node.js | 20+ (Next.js 14 needs ≥18.17) | `node --version` |
| npm | 9+ | `npm --version` |
| Disk | ~1 GB (models + dataset + node_modules) | |

No GPU, no LLM key, no database account is required to run locally: retrieval falls back to an
in-memory cosine index over the TRAIN embeddings.

---

## 2. Install

```bash
git clone https://github.com/Anonymous-Ekansh/NLPSupportSystem.git
cd NLPSupportSystem

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Serving the API (no torch, no training extras)
pip install -r requirements-api.txt

# Only if you will retrain / re-evaluate / rebuild the map
pip install -r requirements-nlp.txt

# UI
cd web && npm install && cd ..
```

The embedding models (`BAAI/bge-small-en-v1.5` for A2 retrieval,
`sentence-transformers/all-MiniLM-L6-v2` for A1) download on first use and are cached by
`fastembed`; check them with:

```bash
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

---

## 3. Environment variables

Copy the template and fill only what you need:

```bash
cp .env.example .env
```

| Variable | Required? | Purpose |
|---|---|---|
| `SUPABASE_URL` | optional | Enables pgvector retrieval instead of the in-memory index |
| `SUPABASE_SERVICE_KEY` | optional | Service-role key — **server-side only**, never in `web/` |
| `ALLOWED_ORIGINS` | optional | CORS for the API (default `*`) |
| `NLP_API_URL` | for the UI | Backend base URL read by `web/` (see `web/.env.local`) |

`web/.env.local`:

```ini
NLP_API_URL=http://localhost:8000
```

There is no LLM key in this system: no model is called at request time. `LLM_API_KEY` in
`.env.example` is unused by Level 2 code.

Verify the install:

```bash
python scripts/check_env.py
```

---

## 4. Data and artifacts that should already exist

| Path | Contents | Regenerate with |
|---|---|---|
| `data/train.csv`, `data/validation.csv` | raw Bitext rows (22,841 total) | — (provided) |
| `outputs/splits/{train,dev,test}.csv` | 15,567 / 1,730 / 3,475 rows | `python -m nlp.split_data` |
| `outputs/models/` | `a1_pipe.joblib`, `a1.npz`, `a1_config.json`, `a2_emb_matrix.joblib`, `a2_train_df.pkl` | `python scripts/train_a1.py` for A1; the A2 matrices are written by `nlp/agent_a2.py` (A2 still works without them — `A2Builder.build()` recomputes the TRAIN embeddings via `nlp/embed.py`) |
| `outputs/evidence/` | `a1.json`, `a2.json`, `day1_baseline_snapshot.json`, `robustness.json`, `l2_decisions.json`, `trace_examples.json` | `scripts/run_all.py`, `scripts/eval_l2.py` |
| `config/l2_policy.json` | all decision thresholds | `scripts/tune_a3_policy.py` (DEV only) |

Full regeneration: `python scripts/run_all.py` (see [scripts/run_all.md](scripts/run_all.md)).

---

## 5. Run the API

```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Docs: <http://localhost:8000/docs>
- Health: `curl -s localhost:8000/health` → reports the five agents, allowed decisions and which
  retrieval backend is live (`pgvector` or `in-memory`).

### Endpoints

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | — | status, agents, decisions, model/backend state |
| `GET /meta` | — | A1 config and measured metrics |
| `POST /resolve` | `{"query": str, "answer": str?, "baseline": bool?}` | decision, final response, predicted intent/category, `a1`–`a5` payloads, `reason_codes`, `evidence_ids`, `trace` / `agent_trace`, `total_latency_ms` |
| `POST /resolve/batch` | `{"queries": [str, …], "baseline": bool?}` | up to 32 results plus decision counts |
| `POST /feedback` | helpfulness vote | acknowledgement |

`query` is limited to 500 characters (HTTP 422 beyond that). `answer` supplies a reply to a
clarification question: the pipeline is stateless and re-runs the original query with the answer
appended, so the intent stays that of the original question while the slot detector sees the new
value.

`baseline: true` restores the Day-1 A1 → A2 path; it keeps every Day-1 response field and
reports `decision: null`.

```bash
curl -s -X POST localhost:8000/resolve -H 'Content-Type: application/json' \
  -d '{"query":"How can I request a refund for my purchase?"}' | python -m json.tool | head -40

curl -s -X POST localhost:8000/resolve -H 'Content-Type: application/json' \
  -d '{"query":"I need a refund but cannot find my order number","answer":"123456"}'
```

---

## 6. Run the CLI

```bash
# one query, human-readable five-agent trace
.venv/bin/python -m nlp.cli "I need a refund but cannot find my order number"

# machine-readable
.venv/bin/python -m nlp.cli --json "where is my order"

# include raw agent payloads
.venv/bin/python -m nlp.cli --payloads "my order never arrived"

# clarification follow-up
.venv/bin/python -m nlp.cli --follow-up "123456" "I need a refund but cannot find my order number"

# Day-1 baseline path
.venv/bin/python -m nlp.cli --baseline "where is my order"

# batch a CSV (column query / instruction / text / message)
.venv/bin/python -m nlp.cli --batch queries.csv \
  --out-csv predictions.csv --out-responses responses.jsonl --out-traces traces.jsonl
```

Exit codes: `0` success, `2` bad input, `1` internal error.

---

## 7. Generate and validate the submission

```bash
# sample the TEST split with a fixed seed and write outputs/submission/
.venv/bin/python scripts/make_submission.py --n 300 --seed 13

# required-column / label / decision validation
.venv/bin/python scripts/validate_submission.py \
  outputs/submission/predictions.csv --gold outputs/submission/gold.csv
```

`predictions.csv` contains **only** `query_id,predicted_intent,predicted_category,decision`.
Responses and traces are written to `responses.jsonl` and `traces.jsonl`, as the rules require.
`submission_report.json` records sampled accuracy, decision counts and latency.

---

## 8. Run the UI

```bash
cd web
npm run dev          # http://localhost:3000
# or a production build
npm run build && npm run start
```

Pages: `/` (live demo + decision timeline), `/results` (held-out metrics), `/architecture`
(agent diagram), `/data` (dataset explorer), `/robustness` (evaluation evidence).

The browser only ever calls `web/app/api/resolve`, which proxies to `NLP_API_URL` server-side. If
the backend is cold or down, the demo scenarios fall back to `web/public/demo_cache.json` and the
UI labels those answers **Cached**.

One-command start (both servers): `./scripts/start_local.sh`

---

## 9. Evaluation and tests

```bash
# held-out robustness evaluation (writes outputs/evidence/*.json; a few minutes on CPU)
.venv/bin/python scripts/eval_l2.py --n 500 --n-pert 100 --seed 13 --sync

# script-style tests: exit code 0 means pass
.venv/bin/python tests/test_a4.py
.venv/bin/python tests/test_pipeline_l2.py

# retune the policy from DEV only (never TEST)
.venv/bin/python scripts/tune_a3_policy.py --dry-run
```

Synchronise evidence to the web app after any regeneration:

```bash
.venv/bin/python scripts/sync_evidence.py
.venv/bin/python scripts/build_demo_cache.py     # refresh cached demo payloads from real runs
```

---

## 10. Optional: Supabase pgvector

1. Create a project, run [`supabase/schema.sql`](supabase/schema.sql) (table `examples`,
   `vector(384)` column, cosine index, `match_examples` RPC, `audit_log` and `metrics`).
2. Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in `.env`.
3. Ingest TRAIN rows: `python scripts/ingest.py` (resumable batches, refuses DEV/TEST rows).
4. Restart the API — `/health` reports `retrieval_backend: "pgvector"`. If the call fails or times
   out, the pipeline silently falls back to the in-memory index and says so in the response.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` | another server on 3000/8000 | `lsof -ti :3000` then stop that PID |
| Answers appear instantly and show **Cached** | backend not reachable from `web/` | start the API, check `NLP_API_URL` in `web/.env.local` |
| `retrieval_backend: "in-memory"` | Supabase unset or unreachable | expected locally; set env vars to use pgvector |
| First request is slow | embedding model download | run once, then it is cached |
| `ModuleNotFoundError: nlp` | wrong working directory | run from the repo root, or set `PYTHONPATH=.` |
| A clarification follow-up escalates | A1 confidence for that phrasing is below `confidence_low` (0.60) | expected safety behaviour; tune the policy on DEV, never TEST |
