# Level 2 Audit — What Day 1 Actually Ships

Scope: read-only audit of the Day 1 code and data before adding A3/A4/A5.
Every claim below cites the file it came from. No source file was modified.

Repo state at audit time: branch `level2`, working tree clean. There was no
prior `docs/L2_AUDIT.md`, so this doc was derived directly from `nlp/`, `api/`,
`scripts/`, `data/` and `outputs/`.

---

## 1. Two parallel A1 / A2 implementations (important)

| Concern | Served path | Legacy / dead path |
|---|---|---|
| A1 | `nlp/a1.py` → `IntentAnalyst` (pure numpy, loads `outputs/models/a1.npz` + `a1_config.json`) | `nlp/agent_a1.py` → `IntentAnalyst` (sklearn TF-IDF + LogReg, `outputs/models/a1_pipe.joblib`) |
| A2 | `nlp/a2.py` → `A2Builder().build()` → `ResponseSpecialist` (retrieval ladder + 3-check self-check) | `nlp/agent_a2.py` → `ResponseSpecialist` (fastembed cosine + shallow self-check) |
| Orchestrator | `nlp/pipeline.py` (`A1Agent`, `A2Agent`, `run_pipeline`) | — |

`api/main.py` startup imports `from nlp.a1 import IntentAnalyst` and
`from nlp.a2 import A2Builder` (api/main.py:76-80), i.e. **the served system is
`a1.py` + `a2.py`**. `scripts/sync_evidence.py` / `build_web_data.py` read the
JSON evidence files. `nlp/agent_a1.py`, `nlp/agent_a2.py`, `nlp/embeddings.py`
are only used by `nlp/examples.py` (an older example generator) — nothing in
`api/` imports them.

Consequence for Level 2: new agents must consume **`nlp.a1.IntentAnalyst.analyze`**
and **`nlp.a2.ResponseSpecialist.respond`**, not the legacy classes.

---

## 2. A1 output contract

`IntentAnalyst.analyze(query) -> dict` (nlp/a1.py, `analyze`, contract also
restated in the module docstring):

| field | type | source / meaning |
|---|---|---|
| `query` | str | the raw input query, unchanged |
| `intent` | str | argmax class name from `a1.npz["classes"]` (27 labels) |
| `category` | str | `intent_to_cat[intent]` — majority category per intent from TRAIN |
| `confidence` | float (4dp) | temperature-scaled softmax max prob |
| `margin` | float (4dp) | `top1 - top2` probability gap |
| `top3` | list[`{intent, prob}`] | top 3 classes, probs 4dp |
| `top1_similarity` | float (4dp) | max cosine to **any TRAIN example embedding** (`train_embs @ q_emb`) |
| `needs_clarification` | bool | `confidence < tau or top1_similarity < s_min` |

Thresholds come from `outputs/models/a1_config.json`:
`served_level="L2"`, `embed_model="sentence-transformers/all-MiniLM-L6-v2"`,
`temperature=0.482836`, `tau=0.441705`, `s_min=0.770013`.
There is **no explicit OOD label**; OOD is expressed only as
`top1_similarity < s_min` (validated in `scripts/train_a1.py` §7 against
`tests/ood_dev.txt`: 96.67 % of OOD flagged, 2.83 % of DEV flagged).
`needs_clarification` is therefore a *combined* low-confidence + OOD signal.

Dev evidence (outputs/metrics/test_metrics.json): test intent accuracy 0.9934,
macro-F1 0.9917, avg confidence 0.9481, `needs_clarification_pct` 0.43 %.
`outputs/evidence/a1.json` gives L2 test accuracy 0.9954 / macro-F1 0.9952,
category accuracy 0.9983, OOD AUROC 0.9643 (max-prob) / 0.9962 (top1-sim).

Risks:
- `a1.npz` / `a1_config.json` must exist at `outputs/models/`; A1 raises at
  import-of-instance time otherwise (`_load_config`).
- In the L2 branch, `analyze()` re-opens `a1.npz` on **every call** just to get
  `train_embs` for `top1_similarity` (nlp/a1.py, `analyze`). Slow but correct;
  Level 2 should not add another full-file load per step.

---

## 3. A2 output contract

`ResponseSpecialist.respond(query, a1_output, mode="hybrid", k=5) -> dict`
(nlp/a2.py, `respond`):

| field | type | meaning |
|---|---|---|
| `response` | str | TRAIN response text of the chosen example (verbatim, may contain `{{...}}` if no candidate passed) |
| `supporting_examples` | list[`{instruction, response, similarity}`] ≤3 | top-ranked TRAIN neighbours |
| `used_intents` | list[str] | distinct intents among the top-k candidates |
| `retrieval_score` | float (4dp) | score of the chosen candidate (hybrid = RRF, dense = cosine) |
| `low_confidence` | bool | echo of A1's `needs_clarification` routing branch |
| `self_check` | dict | see below |

`self_check` = `{placeholder_safe, source_similarity_ok, source_similarity,
response_intent_ok, response_predicted_intent, check_passed,
candidates_rejected}`. Checks are, in order: no `{{` in the response;
`query_emb · source_instruction_emb >= sim_min` (0.9107 by default, chosen on
DEV p5 same-intent similarity); a TF-IDF LogReg trained on TRAIN *responses*
predicts the searched intent. The first passing candidate wins; if none passes
the top-ranked one is returned with `check_passed=False`.

Routing: `needs_clarification=False` → search only top-1 intent;
`True` → search top-3 intents, multiplying each candidate score by A1's prob
for its intent. Fewer than 3 candidates → fall back to full corpus.
Re-ranking puts responses whose placeholders are a subset of the query's
placeholders first (note: the dataset's instructions carry `{{Order Number}}`,
which a real user never types, so this re-rank mostly cannot fire).

Test evidence (`outputs/evidence/a2.json`, mode `dense`): dense_routed
cosine 0.8292 / ROUGE-L 0.3373 / intent-consistency 0.996 /
placeholder-safe 0.7721 / check_passed 0.7085. **A1-misclassified slice
(n=16): intent consistency 0.125** — A2 inherits A1's errors.
`api/main.py::_pgvector_respond` returns a hand-built A2-shaped dict with
`self_check` hard-coded to all-True (api/main.py:135-190). That is a
pre-existing integrity gap: the pgvector path asserts checks it never ran.

---

## 4. Current trace format

`run_pipeline` (nlp/pipeline.py) returns:

```
{ "query": str,
  "trace": [ {"agent": str, "input": {state snapshot before the step},
              "output": {...}, "latency_ms": float} ],
  "final_response": str,          # a2_output.response
  "total_latency_ms": float,
  "map": {"x": float|null, "y": float|null, "confidence": float},
  "retrieval_backend": "pgvector"|"in-memory",
  "planned_agents": [ {name, status, description} x3 ] }
```

`input` is the *entire* prior state dict (`dict(state)` snapshot), so for A2 it
contains `query` + the full `a1_output`. There is **no `role`, `decision`,
`reason` or `evidence` field** in a trace step — Level 2's richer
`TraceStep` must be additive so `web/` and `outputs/examples` keep rendering.
`planned_agents` in nlp/pipeline.py literally names "Verifier", "Clarifier",
"Escalator" as `status: "planned"`; `/meta` returns it verbatim.

pgvector branch in `api/main.py::resolve` builds its own trace inline with
`latency_ms: 0` and no A1 latency (api/main.py:242-257), so trace shapes differ
between backends. Any A3/A4/A5 trace work must tolerate both.

---

## 5. Splits: how they are stored and loaded

Produced by `nlp/split_data.py` (`run_split`) from `data/train.csv` +
`data/validation.csv`:

1. dedupe raw train by `normalize(instruction)`,
2. drop validation rows whose normalised instruction is in train (leak removal),
3. dedupe validation,
4. stratified 90/10 split of deduped train → TRAIN / DEV (`random_state=42`,
   `stratify=intent`),
5. cleaned validation → TEST.

Files on disk: `outputs/splits/train.csv`, `dev.csv`, `test.csv`,
`split_report.json`. `outputs/splits/split_report.json`:

| split | rows | intents | categories | origin |
|---|---|---|---|---|
| train | 15 567 | 27 | 11 | deduped `data/train.csv`, 90 % |
| dev | 1 730 | 27 | 11 | deduped `data/train.csv`, 10 % |
| test | 3 475 | 27 | 11 | cleaned `data/validation.csv` |

Dropped: 1 513 train duplicates, 528 validation leaks, 28 validation duplicates.
Raw inputs: `data/train.csv` 18 810 rows, `data/validation.csv` 4 031 rows.

Loading convention: every consumer reads `outputs/splits/*.csv` with
`pd.read_csv` and applies `nlp.normalize.normalize` to `instruction`
(`nlp/split_data.py`, `nlp/a1.py` docstring/`scripts/train_a1.py`,
`nlp/a2.py::A2Builder.build`). `data/*.csv` is only touched by `split_data.py`
and `scripts/ingest.py`. **There is no test-set leakage in the tuning path**
because `scripts/train_a1.py` and `scripts/eval_a1.py` fit/select on TRAIN and
DEV only; TEST is written once to `outputs/evidence/*.json`.

Note: `outputs/splits/` is **not** committed data — it is regenerated by
`python -m nlp.split_data`; it happens to be present in this checkout.

---

## 6. Dataset `flags` letters (read from the data)

`data/train.csv` + `data/validation.csv` = 22 841 rows; `flags` is a string of
letters (1–9 chars, 371 distinct combinations). Letter counts over all 22 841
rows:

| letter | rows | official meaning (Bitext language-generation tags) | empirical signal in this data |
|---|---|---|---|
| B | 22 841 | Basic syntactic structure | present in **every** row |
| L | 20 472 | Semantic variation (synonyms, hyphens, compounding) | broad, slightly ACCOUNT/ORDER heavy |
| Q | 7 647 | Colloquial variation | even spread across categories |
| I | 6 623 | Interrogative structure | "where can I…", "how do I…" |
| Z | 4 489 | Errors and typos | "temrination", "cancellung" |
| M | 4 194 | Morphological variation | "reimbursements", "notify of problems" |
| C | 2 243 | Coordinated syntactic structure | two-clause requests |
| K | 1 908 | Keyword mode | "refunding {{Currency Symbol}}{{Refund Amount}}", "changkng" |
| W | 1 106 | Offensive language | 100 % profanity rows ("damn", "bloody") |
| P | 1 104 | Politeness variation | "would it be possible to speak to someone" |
| E | 1 618 | Abbreviations ("I'm", "I am") | "I do not know what to do…" |
| N | 391 | Negation | "I can no longer afford purchase…", "can't see…" |
| S | 356 | **not in the official legend** | all 356 rows are DELIVERY/`delivery_options`, phrasing "can i place an order from {{Delivery Country}}" |
| V | 68 | **not in the official legend** | 36 REFUND/`get_refund` + 22 ACCOUNT/`recover_password`, phrasing "I have lost / forgotten … key", "I have paid … I want a refund" |

Official legend source: Bitext dataset README
(https://github.com/bitext/customer-support-llm-chatbot-training-dataset),
section "Language Generation Tags". Letters `D`, `G`, `R`, `Y` are documented
there but **do not occur** in this data; `S` and `V` occur here but are not
documented there — a real inconsistency in the provided files, not a parsing
bug (verified by direct column scan).

Implication for Level 2: `flags` must not be used as a supervision signal for
routing (S/V have no defined meaning, and B is constant). It is useful only as
an analysis slice (`outputs/evidence/a1.json` reports `flag_f1` per letter,
including S/V, which confirms the letters are read as individual characters).

---

## 7. Label sets (exact strings, from the data)

11 categories (as spelled in `data/*.csv` — note these differ from the official
Bitext README names `CANCELLATION_FEE` and `SHIPPING_ADDRESS`; the provided
files use `CANCEL` and `SHIPPING`):

| category | train | dev | test |
|---|---|---|---|
| ACCOUNT | 3 469 | 384 | 748 |
| CANCEL | 594 | 66 | 141 |
| CONTACT | 1 242 | 138 | 291 |
| DELIVERY | 1 056 | 118 | 232 |
| FEEDBACK | 1 256 | 140 | 300 |
| INVOICE | 1 155 | 128 | 260 |
| ORDER | 2 028 | 226 | 402 |
| PAYMENT | 1 253 | 140 | 297 |
| REFUND | 1 659 | 184 | 362 |
| SHIPPING | 1 229 | 136 | 294 |
| SUBSCRIPTION | 626 | 70 | 148 |

27 intents (category = majority category for that intent in TRAIN):

| intent | category | train | dev | test |
|---|---|---|---|---|
| cancel_order | ORDER | 322 | 36 | 41 |
| change_order | ORDER | 557 | 62 | 112 |
| change_shipping_address | SHIPPING | 604 | 67 | 145 |
| check_cancellation_fee | CANCEL | 594 | 66 | 141 |
| check_invoice | INVOICE | 585 | 65 | 137 |
| check_payment_methods | PAYMENT | 627 | 70 | 147 |
| check_refund_policy | REFUND | 618 | 69 | 142 |
| complaint | FEEDBACK | 630 | 70 | 150 |
| contact_customer_service | CONTACT | 624 | 69 | 145 |
| contact_human_agent | CONTACT | 618 | 69 | 146 |
| create_account | ACCOUNT | 576 | 64 | 119 |
| delete_account | ACCOUNT | 580 | 64 | 127 |
| delivery_options | DELIVERY | 431 | 48 | 84 |
| delivery_period | DELIVERY | 625 | 70 | 148 |
| edit_account | ACCOUNT | 514 | 57 | 97 |
| get_invoice | INVOICE | 570 | 63 | 123 |
| get_refund | REFUND | 577 | 64 | 135 |
| newsletter_subscription | SUBSCRIPTION | 626 | 70 | 148 |
| payment_issue | PAYMENT | 626 | 70 | 150 |
| place_order | ORDER | 629 | 70 | 146 |
| recover_password | ACCOUNT | 624 | 69 | 148 |
| registration_problems | ACCOUNT | 624 | 69 | 150 |
| review | FEEDBACK | 626 | 70 | 150 |
| set_up_shipping_address | SHIPPING | 625 | 69 | 149 |
| switch_account | ACCOUNT | 551 | 61 | 107 |
| track_order | ORDER | 520 | 58 | 103 |
| track_refund | REFUND | 464 | 51 | 85 |

Counts are `(df.intent == label).sum()` on the three split CSVs.

---

## 8. `/resolve` request / response schema

Request (`api/main.py::ResolveRequest`, pydantic v2): `{"query": str}`; validator
strips, rejects empty, rejects > 500 chars (`ValueError` → HTTP 422).

Response (`api/main.py::resolve`, in-memory branch = `run_pipeline` output):

```
{
  "query": str,
  "trace": [{"agent": "IntentAnalyst", "input": {...}, "output": {...}, "latency_ms": float},
            {"agent": "ResponseSpecialist", ...}],
  "final_response": str,
  "total_latency_ms": float,
  "map": {"x": float|null, "y": float|null, "confidence": float},
  "retrieval_backend": "in-memory" | "pgvector",
  "planned_agents": [{"name": ..., "status": "planned", "description": ...}]
}
```

Behaviour: per-IP rate limit 30 req / 60 s (HTTP 429); A1 is always run first;
if Supabase is configured the pgvector RPC `match_hybrid` is tried with a 2 s
timeout and falls back to in-memory; every call best-effort inserts into the
Supabase `runs` table. Other endpoints: `GET /health`, `GET /meta`,
`POST /feedback` (`{query, rating 1-5, note?}`).

`/meta` returns `{a1:{served_level, temperature, tau, s_min, n_classes},
retrieval_backend, planned_agents, a1:{...evidence}, a2:{...evidence},
retrieval_backend:{...evidence}}` — the evidence keys come from
`outputs/evidence/{a1,a2,retrieval_backend}.json` if present.

---

## 9. Day 1 behaviours that must stay intact

1. `POST /resolve` still returns `final_response`, `trace`, `map`,
   `retrieval_backend`, `planned_agents` with the **same keys**; `web/` renders
   `map`, the trace timeline and `final_response` from them.
2. A1's 8-field contract and its calibrated `confidence` / `margin` /
   `top1_similarity` / `needs_clarification` semantics (`tau`, `s_min` from
   `a1_config.json`).
3. A2's response text for the same query under the same routing branch
   (`hybrid` default, `k=5`); the 3-stage self-check ordering and
   `candidates_rejected`.
4. `retrieval_backend` honesty: pgvector when the RPC succeeds, in-memory
   otherwise, and the 2 s timeout fallback.
5. The pgvector path must keep working (same A2-shaped dict).
6. 30/60 s rate limiting and the >500-char / empty rejections.
7. `planned_agents` key remains present (the UI/`/meta` show it); Level 2
   replaces its content, it must not 404 the field.
8. No test-set tuning: thresholds are chosen on DEV only.
9. Existing evidence files (`outputs/evidence/a1.json`, `a2.json`,
   `a1_errors.csv`) and `outputs/metrics/test_metrics.json` stay readable by
   `scripts/build_web_data.py` / `sync_evidence.py`.

Regression traps found by reading (do not "clean up" blindly):
- `nlp/examples.py` imports the legacy `nlp.agent_*` classes; it is stale.
- `api/main.py` uses `@app.on_event("startup")` (deprecated in current FastAPI).
- pgvector path fabricates a passing `self_check` (api/main.py:135-190).
- `run_pipeline` derives a trace `output` from `f"{agent.name.lower()}_output"`,
  i.e. `intentanalyst_output` / `responsespecialist_output`, which never exist;
  it only works because of the `IntentAnalyst`/A2 fallback branch. New agents
  must append their own output key or extend this mapping explicitly.
- `outputs/splits/` is generated, not committed; a fresh clone must run
  `nlp/split_data.py` before any model script.

---

## 10. Data facts relevant to missing-slot detection (Level 2)

Derived from TRAIN **instructions** (user-side slots) and TRAIN **responses**
(assistant-side placeholders), per intent (ratio of rows containing it):

- Strongly required, user-side: `cancel_order` Order Number 310/322 (.96),
  `change_order` 537/557 (.96), `track_order` 504/520 (.97),
  `check_invoice` Person Name 226/585 (.39), `get_invoice` Person Name
  240/570 (.42), `create/delete/edit/switch_account` Account Type ≈ .22–.28.
- Assistant-side only (response placeholders, not user input):
  `delivery_period` Order Number .91 / Tracking Number .89,
  `track_refund` Refund Amount .19, `get_refund` Refund Amount .12,
  `cancel_order` Customer Support Phone/Hours .91.
- Responses contain 200+ distinct placeholder strings, including broken ones
  (`Choose the {{Account Type`, `customer_support_phone_number`) — placeholder
  parsing must be tolerant.

So a purely data-driven "required slot" list will not by itself produce the
required Level 2 behaviour ("I need a refund but cannot find my order number"
must ask for details): `get_refund` has no dominant user-side slot. The policy
file therefore needs an explicit intent→required-slot override map, seeded from
this evidence, with the data ratios kept as the documented provenance.

---

## 11. Baseline snapshot

`outputs/evidence/day1_baseline_snapshot.json` records the Day 1 headline
numbers (A1 ladder + test metrics, A2 retrieval modes, split sizes, label
inventory) copied verbatim from `outputs/evidence/a1.json`,
`outputs/evidence/a2.json`, `outputs/metrics/test_metrics.json`,
`outputs/models/a1_config.json`, `outputs/splits/split_report.json`.
