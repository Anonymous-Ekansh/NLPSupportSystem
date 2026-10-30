# A1 Intent Analyst — Evaluation Results

**Served model:** L2 (embedding → LogReg (C=50.0))

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

**Temperature:** 0.4828  |  **TAU:** 0.4417  |  **S_MIN:** 0.7700

**Test set:** 3475 queries


## Embedding Bake-off

| Model | DEV log-loss | DEV accuracy | DEV Macro-F1 |
|-------|-------------|-------------|-------------|
| `BAAI/bge-small-en-v1.5` | 0.151573 | 0.9931 | 0.9916 |
| `sentence-transformers/all-MiniLM-L6-v2` ★ | 0.100246 | 0.9919 | 0.9910 |

## Model Ladder (TEST)

| Level | Name | Accuracy | Macro-F1 | 95% CI | Cat Acc |
|-------|------|----------|----------|--------|---------|
| L0 | majority class | 0.0432 | 0.0031 | [0.0026, 0.0035] | 0.0863 |
| L1 | TF-IDF word+char_wb → LogReg | 0.9991 | 0.9988 | [0.9973, 1.0000] | 1.0000 |
| L2 ★ | embedding → LogReg (C=50.0) | 0.9954 | 0.9952 | [0.9927, 0.9974] | 0.9983 |
| L3 | sim-weighted kNN (k=15, T=0.05) | 0.9928 | 0.9913 | [0.9873, 0.9947] | 0.9991 |

**Paired bootstrap (L2 − L1):** ΔF1 95% CI = [-0.0061, -0.0015]
→ CI **excludes zero**: the served model is significantly **worse** than L1.


## Per-Intent F1 (served model)

| Intent | F1 |
|--------|------|
| cancel_order | 0.9877 |
| change_order | 0.9955 |
| change_shipping_address | 0.9896 |
| check_cancellation_fee | 1.0000 |
| check_invoice | 0.9963 |
| check_payment_methods | 0.9966 |
| check_refund_policy | 0.9930 |
| complaint | 0.9933 |
| contact_customer_service | 0.9965 |
| contact_human_agent | 0.9966 |
| create_account | 0.9917 |
| delete_account | 1.0000 |
| delivery_options | 0.9882 |
| delivery_period | 0.9966 |
| edit_account | 1.0000 |
| get_invoice | 0.9960 |
| get_refund | 0.9851 |
| newsletter_subscription | 1.0000 |
| payment_issue | 0.9933 |
| place_order | 0.9966 |
| recover_password | 1.0000 |
| registration_problems | 0.9933 |
| review | 1.0000 |
| set_up_shipping_address | 0.9933 |
| switch_account | 0.9953 |
| track_order | 0.9952 |
| track_refund | 1.0000 |

## Top-10 Confusions

| True | Predicted | Count |
|------|-----------|-------|
| get_refund | check_refund_policy | 2 |
| switch_account | create_account | 1 |
| set_up_shipping_address | change_shipping_address | 1 |
| registration_problems | create_account | 1 |
| payment_issue | registration_problems | 1 |
| payment_issue | get_refund | 1 |
| get_refund | check_payment_methods | 1 |
| delivery_period | delivery_options | 1 |
| contact_customer_service | contact_human_agent | 1 |
| complaint | place_order | 1 |

## Macro-F1 per Flag Letter

| Flag | Macro-F1 | n |
|------|----------|---|
| B | 0.9952 | 3475 |
| C | 0.9928 | 362 |
| E | 0.9935 | 242 |
| I | 0.9928 | 1094 |
| K | 0.9970 | 198 |
| L | 0.9952 | 3198 |
| M | 0.8911 | 671 |
| N | 1.0000 | 53 |
| P | 0.9900 | 186 |
| Q | 0.9933 | 1245 |
| S | 1.0000 | 2 |
| V | 1.0000 | 16 |
| W | 0.9856 | 193 |
| Z | 0.9707 | 685 |

## Calibration

| | ECE |
|--|-----|
| Before temperature scaling | 0.0082 |
| After temperature scaling (T=0.4828) | 0.0027 |

![Reliability Diagram](outputs/figures/a1_reliability.png)


## Risk-Coverage

![Risk-Coverage Curve](outputs/figures/a1_risk_coverage.png)


## OOD Detection

| Detector | AUROC |
|----------|-------|
| max-prob | 0.9643 |
| top1_similarity | 0.9962 |

> ⚠ `ood_dev.txt` informed `S_MIN`, so this AUROC is **optimistic**.


## Thresholds

- **TAU** = 0.4417 (coverage = 100.00%)
- **S_MIN** = 0.7700 (DEV flagged = 2.83%, OOD flagged = 96.67%)

## Errors

Total errors: **16** / 3475 (0.46%)

See `outputs/evidence/a1_errors.csv` for full error list.
