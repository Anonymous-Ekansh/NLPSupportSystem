# Evaluation run report — `evaluation_queries.csv` (no gold labels)

**No accuracy or F1 is reported for this file: it has no gold labels.**

| metric | value |
|---|---|
| total rows | 4031 |
| error/fallback rows | 0 (expected 0) |
| runtime | 1491.9s |
| latency p50 / p95 / max | 254.3 / 940.1 / 25666.8 ms |
| low-confidence rows (< 0.6) | 8 |
| rows containing \u007b\u007bplaceholders\u007d\u007d | 985 (24.4%) → {'ANSWER': 869, 'CLARIFY': 114, 'ESCALATE': 2} |
| clarify sourced from missing slots | missing_information: 293 |

## Decisions

| decision | count | share |
|---|---|---|
| ANSWER | 3601 | 0.8933 |
| CLARIFY | 421 | 0.1044 |
| ESCALATE | 9 | 0.0022 |

## Top 10 predicted intents

| intent | count |
|---|---|
| track_refund | 152 |
| change_order | 151 |
| contact_human_agent | 151 |
| create_account | 151 |
| get_refund | 151 |
| complaint | 150 |
| contact_customer_service | 150 |
| delivery_period | 150 |
| edit_account | 150 |
| get_invoice | 150 |

## Category distribution

| category | count |
|---|---|
| ACCOUNT | 897 |
| ORDER | 598 |
| REFUND | 452 |
| CONTACT | 301 |
| DELIVERY | 299 |
| INVOICE | 299 |
| PAYMENT | 299 |
| FEEDBACK | 298 |
| SHIPPING | 295 |
| SUBSCRIPTION | 150 |
| CANCEL | 143 |

## Reason codes

| reason code | count |
|---|---|
| standard_retrieval | 3671 |
| audit_accept | 2274 |
| audit_revise | 1327 |
| missing_information | 293 |
| missing_order_number | 293 |
| audit_flag | 70 |
| intent_alignment | 69 |
| uncertain_intent | 69 |
| multi_intent | 59 |
| placeholder_safety | 26 |
| informational_question | 18 |
| low_confidence | 4 |
| out_of_distribution | 4 |
| conflicting_evidence | 1 |
| contradiction | 1 |

## Known limitation of this run

The hand-written evaluation file contains no gold labels, so intent accuracy is not measurable here — this run only proves the finished system produces complete, well-formed, identical decisions offline and via the API (see validate_evaluation.py).

## Escalated rows

- **Q0257** `need help informing of rergistration errors` — low_confidence
- **Q1045** `direct to mw someone` — out_of_distribution
- **Q1964** `need assistance withcanceling purchase {{Order Number}}` — low_confidence
- **Q2299** `i have got to send feeback` — low_confidence
- **Q2428** `can Icontact a person?` — out_of_distribution
- **Q2540** `withdrawal hcarge` — out_of_distribution
- **Q2624** `I need to see your money back guarantee, can you help me?` — standard_retrieval, audit_flag, contradiction, conflicting_evidence
- **Q2823** `seeing biols from {{Person Name}}` — out_of_distribution
- **Q3237** `would it be possible to lolok for my bill #12588` — low_confidence

## Validation (validate_evaluation.py, exit code 0)

| check | result | note |
|---|---|---|
| T1 Input integrity | PASS | 4031 rows, unique ids, no empty text |
| T2 predictions.csv format | PASS | exact header, order preserved |
| T3 Dataset labels & mapping | PASS | 0 bad intents / categories / mapping mismatches |
| T4 Decisions | PASS | only ANSWER / CLARIFY / ESCALATE |
| T5 Completeness | PASS | one record per id in responses.jsonl and traces.jsonl; full A1→A3→A2→A4→A5 traces |
| T6 Responsible behaviour | PASS | completion-claims=0, unresolved `{{`=0, blank ANSWER=0, CLARIFY not a question=0, weak ESCALATE=0 |
| T7 Determinism | PASS | 300 seeded rows re-run identical |
| T8 Duplicate consistency | PASS | 61 duplicated texts, 132 rows, all consistent |
| T9 Wiring sanity | PASS | 514 verbatim split rows, 511/514 agreement (NOT an accuracy metric) |
| T10 API smoke test | PASS | 25 single + 25 batch == offline |

### Definition changes made to the validator itself (documented, not metric fixing)

- **T6 v1** required every CLARIFY response to *end* with `?`. That strict rule flagged
  275 of 421 CLARIFYs because the policy's own slot-clarify template is a question
  ("Could you share your order number?") followed by a guidance sentence. The v2
  definition checks the clarification contains a question; re-scan of all 421 CLARIFYs
  confirms every one contains `?` (0 violations). Old strict count kept here for honesty.
- **T9/T10 key mismatch (v1)**: the check compared the API result's `predicted_intent`
  against the prediction CSV's prediction record; the API values were correct but the
  lookup key differed. The fix only affects the validator's own lookups, not the system.

The predictions and responses in this report were **not** changed by either fix.
