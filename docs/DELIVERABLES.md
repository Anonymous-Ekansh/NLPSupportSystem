# Deliverables checklist (Level 2 problem statement)

Each row names a requirement from the brief, the file or command that satisfies it, and how to
verify it. Nothing here is aspirational: every path exists in this repository.

## Agents

| Requirement | Implementation | Verify |
|---|---|---|
| A1 Intent Analyst (retained) | [`nlp/a1.py`](../nlp/a1.py) | `curl -s localhost:8000/meta` |
| A2 Response Specialist (retained) | [`nlp/a2.py`](../nlp/a2.py) | expand A2 in the UI trace |
| **A3 Routing / Planning** — next steps from intent, uncertainty, evidence | [`nlp/l2/a3.py`](../nlp/l2/a3.py) | trace step 2 shows `route`, reasons and reason codes |
| **A4 Response Auditor** — alignment, grounding, unsupported claims; accept / revise / flag | [`nlp/l2/a4.py`](../nlp/l2/a4.py) | trace step 4 lists all five checks and the verdict |
| **A5 Coordinator** — one decision + agent trace | [`nlp/a5.py`](../nlp/a5.py) | `decision` in every `/resolve` response |
| Shared contracts | [`nlp/l2/contracts.py`](../nlp/l2/contracts.py) | imported by all three agents |
| Tunable policy | [`config/l2_policy.json`](../config/l2_policy.json) | [`docs/DECISION_POLICY.md`](DECISION_POLICY.md) |

## Decisions and responsible behaviour

| Requirement | Implementation | Verify |
|---|---|---|
| ANSWER only with relevant evidence and a checked response | [`nlp/a5.py`](../nlp/a5.py) `coordinate()` | A5 refuses unaudited candidates (`unchecked_candidate`) |
| CLARIFY for ambiguity or missing essentials, one focused question | A3 rule 4/6 + A5 `_clarify_response` | `python -m nlp.cli "I need a refund but cannot find my order number"` |
| ESCALATE for unsupported / low confidence / unresolved conflict | A3 rules 1–3, 5 + A5 flag mapping | `python -m nlp.cli "I will sue you over this charge"` |
| Never claim an order/payment/refund/account action happened | A4 `unsupported_claims` + deterministic rewrite | [`tests/test_a4.py`](../tests/test_a4.py) property sweep |
| Agent trace with role, input, output, decision, evidence, latency | `TraceStep` in [`nlp/l2/contracts.py`](../nlp/l2/contracts.py), rendered by `web/components/decisions/AgentTimeline.tsx` | any `/resolve` response → `trace`, `agent_trace` |

## Robustness testing and results

| Requirement | Implementation | Verify |
|---|---|---|
| Spelling errors, abbreviations, colloquial phrasing | [`nlp/perturb.py`](../nlp/perturb.py) | `python -c "import nlp.perturb"` |
| Dataset `flags` variation groups | `nlp/perturb.py` flag meanings + `scripts/eval_l2.py` per-flag slices | [`docs/L2_AUDIT.md`](L2_AUDIT.md) §6 |
| Results by variation type with sample sizes | [`scripts/eval_l2.py`](../scripts/eval_l2.py) → `outputs/evidence/robustness.json` | open **/robustness** in the UI |
| Uncertain intent, contradictory evidence, weak evidence → revise / clarify / escalate | stress sets in `eval_l2.py`; curated traces in `outputs/evidence/trace_examples.json` | `/robustness` stress table, and A4 BEFORE/AFTER in the trace |
| Intent/category metrics on held-out data | `robustness.json` → `overall`, matched against `outputs/evidence/a1.json` | **/results** page |
| Evidence and decision traces shown | `outputs/evidence/*.json` synced to `web/public/evidence/` | `/robustness`, timeline drawer |

## Application and submission

| Requirement | Implementation | Verify |
|---|---|---|
| Working end-to-end application | API [`api/main.py`](../api/main.py), CLI [`nlp/cli.py`](../nlp/cli.py), UI [`web/`](../web) | `./scripts/start_local.sh` |
| Runnable code and dependencies | [`requirements-api.txt`](../requirements-api.txt), [`requirements-nlp.txt`](../requirements-nlp.txt), [`web/package.json`](../web/package.json) | `pip install -r requirements-api.txt` |
| Setup guide | [`SETUP.md`](../SETUP.md) | follow §2–§8 |
| Short demonstration | [`docs/DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | run it in ~5 minutes |
| Intent/category predictions | `outputs/submission/predictions.csv` | [`scripts/make_submission.py`](../scripts/make_submission.py) |
| Generated/selected responses, separate from the CSV | `outputs/submission/responses.jsonl` | one JSON object per query id |
| Agent traces, separate from the CSV | `outputs/submission/traces.jsonl` | contains roles, inputs, outputs, decisions, evidence, latency |
| CSV columns exactly `query_id,predicted_intent,predicted_category,decision` | `scripts/make_submission.py` | `python scripts/validate_submission.py outputs/submission/predictions.csv` |
| Exact dataset labels | enforced by the validator against `config/l2_policy.json` + splits | same command |
| Decisions limited to ANSWER / CLARIFY / ESCALATE | enforced by the validator | same command |

## Reproduction

| Artifact | Command |
|---|---|
| Splits (15,567 / 1,730 / 3,475) | `python -m nlp.split_data` |
| A1 model + calibration | `python scripts/train_a1.py` |
| A1 / A2 evidence | `python scripts/eval_a1.py`, `python scripts/eval_a2.py` |
| Policy (DEV only) | `python scripts/tune_a3_policy.py` |
| Level-2 robustness evidence | `python scripts/eval_l2.py --sync` |
| Cached demo payloads | `python scripts/build_demo_cache.py` |
| Submission files | `python scripts/make_submission.py --n 300 --seed 13` |
| Test suites | `python tests/test_a4.py`, `python tests/test_pipeline_l2.py` |

## Not in this submission

- No LLM generation: answers are retrieved and audited, never composed by a model.
- No real account actions: the system has no write access to any order, payment or refund system.
- No real customer traffic: robustness perturbations are synthetic and labelled as such.
