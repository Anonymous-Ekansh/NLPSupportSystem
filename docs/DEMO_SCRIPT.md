# 5-minute demo script

Everything below runs locally with `./scripts/start_local.sh` (API `:8000`, UI `:3000`).
Have two terminal tabs open: one for the UI, one for the CLI. Type the queries live where
possible — the judges should see that nothing is scripted.

---

## 0:00 — Frame it (30 s)

> "Five agents, one decision. A1 reads the question, A3 decides whether answering is even
> allowed, A2 retrieves a real past reply as evidence, A4 checks that reply for grounding and
> unsupported claims, and A5 commits to ANSWER, CLARIFY or ESCALATE — with the whole trace.
> No LLM is called at request time and nothing is ever executed on a customer account."

Open <http://localhost:3000>. Point at the seven one-click scenarios and the `A1 → A3 → A2 → A4 → A5`
strip.

---

## 0:30 — The happy path (60 s)

**Type:** `How can I request a refund for my purchase?`

- Expect the badge **ANSWER**.
- Click **How I answered this**.
- Walk the timeline top to bottom: A1 (intent + confidence + similarity), A3 (`route=answer`,
  reason: an informational question needs no order number), A2 (retrieved candidate +
  supporting examples), A4 (`verdict: accept`, five checks), A5 (final decision).
- Note the per-check lines under A4: intent alignment with the cosine, `n/n` sentences grounded,
  `0` completion claims, no unresolved placeholders, neighbour agreement.

**Say:** "A4 is the reason we can answer at all — the text that ships is the audited text."

---

## 1:30 — The case from the problem statement (60 s)

**Type:** `I need a refund but cannot find my order number`

- Expect **CLARIFY**, not "your refund has been processed".
- Read the question aloud: it asks for the order number *and* offers a fallback (use the email
  address instead) — exactly the behaviour the brief requires.
- Expand the trace: A3 shows `missing_information` + `user_lacks_slot`; A2 and A4 are marked
  skipped, and the trace still contains all five steps.

**Type a follow-up** in the reply box: `123456`

- The pipeline re-runs the original question with the answer appended (stateless).
- Watch the badge: this phrasing sits at confidence ≈0.56, below the 0.60 escalation floor, so it
  hands off to a human with `low_confidence` and an explicit "no changes have been made".
  Say so plainly — it is the safety rule doing its job, not a hidden failure.

---

## 2:30 — Ambiguity, noise and abuse (90 s)

Three quick typed queries:

| Type this | Expect | Point out |
|---|---|---|
| `refund` | **CLARIFY** with options | One word is not enough; A5 asks "Are you asking about X or Y?" and also names the missing order number |
| `asdkjhqwoieu 🎉🎉` | **ESCALATE** | Gibberish never reaches retrieval |
| `I will sue you over this charge` | **ESCALATE** | Sensitive-content rule, queue `legal-escalations`, reason `sensitive_legal_threat` |

Then click the **Hinglish** scenario (`bhai mera refund kab aayega please jaldi batao`) and expand
A4 — this is where a **revise** is visible: show the BEFORE (A2 candidate) and AFTER (A4 rewrite)
text side by side.

---

## 4:00 — The evidence behind it (45 s)

Open **/robustness**:

- The held-out sample size, seed and policy file are printed at the top, together with the Day-1
  reference the sample is compared against.
- The variation table lists dataset `flags` groups and synthetic perturbations with `n` per group;
  groups under 30 are tagged **low sample** — say that out loud.
- Decision distribution, coverage and accuracy-on-answered sit next to the risk–coverage curve.
- The stress table is the honest part: ambiguous blends and weak-evidence rows are still mostly
  answered, and the page says so. Compare with the Day-1 column, which always answers.
- Close on **0 false action claims** in the checked outputs.

Then **/results** (Day-1 intent/category metrics, unchanged) and **/architecture** (the agent
handoff) if time allows.

---

## 4:45 — Close (15 s)

> "One CSV with the four required columns, responses and traces as separate files, every decision
> reproducible from a seeded split, and a policy file that was tuned on validation only."

Hand over `outputs/submission/` and mention:

```bash
.venv/bin/python scripts/validate_submission.py outputs/submission/predictions.csv \
  --gold outputs/submission/gold.csv
```

---

## If a judge types something unexpected

| Situation | What to do |
|---|---|
| A query answers with a slightly odd dataset-style sentence | Expected: answers are retrieved from a templated corpus, not generated. Point at the evidence block. |
| A query escalates that "should" have answered | Open A3's reason line — it names the rule (`low_confidence`, `out_of_distribution`, …). That is the policy, not a bug. |
| The page shows **Cached** | The backend is down or cold; say so, and restart it (`/health` should report `level2: true`). |
| Backend takes seconds on the first query | Embedding model warm-up; subsequent requests are fast. |

---

## Do not claim

- That any order, refund, payment or account was changed — the system cannot do that.
- That the robustness perturbations are real customer traffic — they are synthetic and labelled.
- Numbers that are still `PENDING` in the docs — read them from `outputs/evidence/` live, or say
  the final evaluation run is pending.
