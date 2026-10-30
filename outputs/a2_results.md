# A2 Response Specialist — Evaluation Results

**Served mode:** `dense` with A1 routing

**Test set:** 3475 queries

**A1 accuracy on TEST:** 0.9954

**A1 low-confidence rate:** 3.63%


## Retrieval Mode Comparison (TEST)

| Mode | Cosine Sim | ROUGE-L F1 | Intent Cons. | PH-Safe | Check-Pass | Check-Reject |
|------|-----------|-----------|-------------|---------|-----------|-------------|
| sparse_routed | 0.8131 | 0.3285 | 0.9963 | 0.7770 | 0.6547 | 0.4532 |
| sparse_global | 0.8060 | 0.3368 | 0.9960 | 0.5819 | 1.0000 | 0.0000 |
| dense_routed ★ | 0.8292 | 0.3373 | 0.9960 | 0.7721 | 0.7085 | 0.2990 |
| dense_global | 0.8394 | 0.3499 | 0.9983 | 0.5833 | 1.0000 | 0.0000 |
| hybrid_routed | 0.8242 | 0.3318 | 0.9960 | 0.7758 | 0.6967 | 0.3312 |
| hybrid_global | 0.8366 | 0.3449 | 0.9994 | 0.5862 | 1.0000 | 0.0000 |
| random_routed | 0.6769 | 0.2876 | 1.0000 | 0.5799 | 1.0000 | 0.0000 |

## Analysis

- **sparse**: routing vs global difference is **tiny** (ΔROUGE-L = -0.0083, Δcosine = +0.0071). A1 is so accurate that filtering to the right intent offers marginal gains.
- **dense**: routing hurts substantially (ΔROUGE-L = -0.0126, Δcosine = -0.0102).
- **hybrid**: routing hurts substantially (ΔROUGE-L = -0.0131, Δcosine = -0.0124).

- **Random baseline** (same-intent): ROUGE-L = 0.2876 vs served 0.3373. Gap is modest — many same-intent responses are similar.

## Slices

### A1-Misclassified Queries

- **sparse_routed**: n=16, cosine=0.5943, ROUGE-L=0.1909, intent_cons=0.1875
- **sparse_global**: n=16, cosine=0.7229, ROUGE-L=0.3178, intent_cons=0.9375
- **dense_routed**: n=16, cosine=0.5612, ROUGE-L=0.1845, intent_cons=0.1250
- **dense_global**: n=16, cosine=0.6805, ROUGE-L=0.2793, intent_cons=0.8125
- **hybrid_routed**: n=16, cosine=0.5963, ROUGE-L=0.1886, intent_cons=0.1250
- **hybrid_global**: n=16, cosine=0.6889, ROUGE-L=0.2898, intent_cons=0.9375

### Low-Confidence Queries

- **sparse_routed**: n=126, cosine=0.7407, ROUGE-L=0.3062, intent_cons=0.9603
- **sparse_global**: n=126, cosine=0.7283, ROUGE-L=0.3149, intent_cons=0.9762
- **dense_routed**: n=126, cosine=0.7485, ROUGE-L=0.3050, intent_cons=0.9524
- **dense_global**: n=126, cosine=0.7500, ROUGE-L=0.3094, intent_cons=0.9603
- **hybrid_routed**: n=126, cosine=0.7486, ROUGE-L=0.3043, intent_cons=0.9524
- **hybrid_global**: n=126, cosine=0.7554, ROUGE-L=0.3118, intent_cons=0.9921

## Mode Selection (DEV ROUGE-L)

| Mode | DEV ROUGE-L |
|------|------------|
| sparse | 0.3315 |
| dense ★ | 0.3364 |
| hybrid | 0.3358 |
