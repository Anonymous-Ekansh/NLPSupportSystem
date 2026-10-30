"""A4 auditor contract tests + 500-row TRAIN property sweep.

Run:  .venv/bin/python tests/test_a4.py   (exit 0 = all pass)

Covers: verdicts accept / revise / flag; checks (a) intent alignment,
(b) grounding, (c) unsupported claims + invented specifics, (d) placeholder
safety, (e) contradiction; the "refund processed" case; the retry loop.
Property sweep samples 500 TRAIN responses through the whole audit and
asserts no accepted/revised output contains a completion claim or an
unresolved {{...}} placeholder, and prints the verdict tally.
"""
from __future__ import annotations

import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from nlp.l2.a4 import (audit, audit_with_retry, build_retry_candidate,
                       claim_matches, invented_specifics, strip_placeholders)
from nlp.l2.contracts import RoutePlan
from nlp.l2.policy import load_policy

POLICY = load_policy()
INTENT2CAT = POLICY["intents"]
PLAN = RoutePlan(route="answer")

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: object = "") -> None:
    if cond:
        print(f"  ok  {name}")
    else:
        print(f"FAIL  {name}  {detail!r}")
        FAILURES.append(name)


def a1(query: str, intent: str, clar: bool = False) -> dict:
    return {
        "query": query, "intent": intent,
        "category": INTENT2CAT.get(intent, ""),
        "confidence": 0.95, "needs_clarification": clar,
        "top3": [{"intent": intent, "prob": 0.95}],
        "top1_similarity": 0.95, "margin": 0.4,
    }


def a2(response: str, evid: list[tuple[str, str, str]]) -> dict:
    return {
        "response": response,
        "supporting_examples": [
            {"instruction": i, "response": r, "similarity": 0.92, "intent": t}
            for (i, r, t) in evid
        ],
        "used_intents": sorted({t for (_, _, t) in evid}),
        "retrieval_score": 0.9, "low_confidence": False,
        "self_check": {"check_passed": True},
    }


train = pd.read_csv("outputs/splits/train.csv")


def clean_fixture(intent: str, need: int = 2) -> dict:
    """First TRAIN row of *intent* whose response is unambiguous for A4
    (no placeholders, no completion claims, no invented specifics)."""
    for _, row in train[train.intent == intent].iterrows():
        resp = str(row.response)
        bad = ("{{" in resp or "}}" in resp
               or claim_matches(resp) or invented_specifics(resp, ""))
        if not bad:
            return {"instruction": str(row.instruction),
                    "response": resp, "intent": intent}
        need_need = True  # keep scanning
    raise RuntimeError(f"no clean TRAIN row found for intent {intent!r}")


def evidence_of(fx: dict) -> list:
    """Single same-intent evidence for *fixture*."""
    return [(fx["instruction"], fx["response"], fx["intent"])]


def run_case(name: str, fixture: dict, candidate: str):
    """Audit *candidate* against same-intent evidence from *fixture*."""
    res = audit(fixture["instruction"], a1(fixture["instruction"],
                                           fixture["intent"]),
                PLAN, a2(candidate, evidence_of(fixture)))
    return res


print("── fixtures (real TRAIN rows) ──")
FX_POLICY = clean_fixture("check_refund_policy")
FX_ORDER = clean_fixture("track_order")
FX_TRACK_REFUND = clean_fixture("track_refund")
FX_CANCEL = clean_fixture("cancel_order")
FX_DELIVERY = clean_fixture("delivery_period")
for label, fx in [("check_refund_policy", FX_POLICY), ("track_order", FX_ORDER),
                  ("track_refund", FX_TRACK_REFUND),
                  ("cancel_order", FX_CANCEL), ("delivery_period", FX_DELIVERY)]:
    print(f"  {label}: intent={fx['intent']} response[:60]={fx['response'][:60]!r}")

print("── case 1: accept, clean TRAIN-sourced candidate ──")
res = run_case("c1", FX_POLICY, FX_POLICY["response"])
check("c1 verdict accept", res.verdict == "accept", res.to_dict())
check("c1 no issues", not res.issues, res.issues)

print("── case 2: accept, second intent family ──")
res2 = run_case("c2", FX_ORDER, FX_ORDER["response"])
check("c2 verdict accept", res2.verdict == "accept", res2.to_dict())

print("── case 3: 'refund processed' passive claim → revise ──")
res3 = run_case("c3", FX_TRACK_REFUND,
                "Your refund has been processed. " + FX_TRACK_REFUND["response"])
check("c3 verdict revise", res3.verdict == "revise", res3.to_dict())
chk3 = {c.name: c.passed for c in res3.checks}
check("c3 unsupported_claims failed", chk3.get("unsupported_claims") is False,
      [(c.name, c.passed) for c in res3.checks])
rewritten = res3.revised_response or ""
check("c3 revised has no claim", not claim_matches(rewritten), rewritten)
check("c3 revised keeps safe tail",
      FX_TRACK_REFUND["response"].split(".")[0][:25].lower()
      in rewritten.lower(), rewritten)
check("c3 revised softens claim", "can be processed" in rewritten, rewritten)

print("── case 4: 'I have cancelled' active claim → revise ──")
res4 = run_case("c4", FX_CANCEL,
                "I have cancelled your order. " + FX_CANCEL["response"])
check("c4 verdict revise", res4.verdict == "revise", res4.to_dict())
check("c4 revised drops active claim",
      "have cancelled" not in (res4.revised_response or ""),
      res4.revised_response)

print("── case 5: 'refund is on its way' promise → revise ──")
res5 = run_case("c5", FX_TRACK_REFUND,
                "Your refund is on its way. " + FX_TRACK_REFUND["response"])
check("c5 verdict revise", res5.verdict == "revise", res5.to_dict())
check("c5 revised drops promise",
      "on its way" not in (res5.revised_response or ""), res5.revised_response)
check("c5 revised keeps safe tail",
      FX_TRACK_REFUND["response"].split(".")[0][:25].lower()
      in (res5.revised_response or "").lower(), res5.revised_response)

print("── case 6: unresolved {{Order Number}} → revise with neutral wording ──")
res6 = run_case("c6", FX_CANCEL,
                "We can cancel order {{Order Number}} for you. "
                + FX_CANCEL["response"])
check("c6 verdict revise", res6.verdict == "revise", res6.to_dict())
chk6 = {c.name: c.passed for c in res6.checks}
check("c6 placeholder check failed", chk6.get("placeholder_safety") is False,
      [(c.name, c.passed) for c in res6.checks])
rev6 = res6.revised_response or ""
check("c6 revised neutral wording", "{{" not in rev6, rev6)
check("c6 revised mentions order number", "order number" in rev6, rev6)

print("── case 7: invented amount → revise, value removed ──")
res7 = run_case("c7", FX_TRACK_REFUND,
                "A refund of $49.99 has been issued. "
                + FX_TRACK_REFUND["response"])
check("c7 verdict revise", res7.verdict == "revise", res7.to_dict())
check("c7 invented amount gone", "$49.99" not in (res7.revised_response or ""),
      res7.revised_response)
check("c7 invented reported", any("invented" in i for i in res7.issues),
      res7.issues)

print("── case 8: invented date → revise ──")
res8 = run_case("c8", FX_DELIVERY,
                "Your replacement was shipped on December 5 and shipped "
                "again anyway. " + FX_DELIVERY["response"])
check("c8 verdict revise", res8.verdict == "revise", res8.to_dict())
rev8 = res8.revised_response or ""
check("c8 date sentence dropped", "december" not in rev8.lower(), rev8)
check("c8 safe tail kept",
      FX_DELIVERY["response"].split(".")[0][:25].lower() in rev8.lower(), rev8)

print("── case 9: whole response is a claim → flag ──")
res9 = run_case("c9", FX_TRACK_REFUND,
                "We have issued the refund to your card.")
check("c9 verdict flag", res9.verdict == "flag", res9.to_dict())

print("── case 10: misaligned evidence intent → flag + retry ──")
pred_intent = FX_ORDER["intent"]
fx_inactive = FX_POLICY  # different intent tag on purpose
res10 = audit(FX_ORDER["instruction"], a1(FX_ORDER["instruction"], pred_intent),
              PLAN,
              a2(FX_POLICY["response"],
                 evidence_of(fx_inactive) + evidence_of(FX_ORDER)))
check("c10 verdict flag", res10.verdict == "flag", res10.to_dict())
check("c10 retry_candidate_rank=2", res10.retry_candidate_rank == 2,
      res10.retry_candidate_rank)
chk10 = {c.name: c.passed for c in res10.checks}
check("c10 alignment check failed", chk10.get("intent_alignment") is False,
      [(c.name, c.passed) for c in res10.checks])

print("── case 11: misaligned with single evidence → flag, no retry ──")
res11 = audit(FX_ORDER["instruction"],
              a1(FX_ORDER["instruction"], FX_ORDER["intent"]),
              PLAN, a2(FX_POLICY["response"], evidence_of(FX_POLICY)))
check("c11 verdict flag", res11.verdict == "flag", res11.to_dict())
check("c11 no retry offered", res11.retry_candidate_rank is None,
      res11.retry_candidate_rank)

print("── case 12: neighbour intent disagreement → flag (contradiction) ──")
refund_fx = clean_fixture("get_refund")
pw_fx = clean_fixture("recover_password")
res12 = audit(refund_fx["instruction"], a1(refund_fx["instruction"],
                                           "get_refund"),
              PLAN,
              a2("You can request a refund within 30 days of purchase.",
                 evidence_of(refund_fx) + evidence_of(pw_fx)))
check("c12 verdict flag", res12.verdict == "flag", res12.to_dict())
chk12 = {c.name: c.passed for c in res12.checks}
check("c12 contradiction check failed", chk12.get("contradiction") is False,
      [(c.name, c.passed) for c in res12.checks])

print("── case 13: ungrounded sentence, single candidate → revise ──")
res13 = run_case("c13", FX_ORDER,
                 FX_ORDER["response"]
                 + " The moon is rather far away tonight entirely unseen.")
check("c13 verdict revise", res13.verdict == "revise", res13.to_dict())
check("c13 ungrounded dropped", "moon" not in (res13.revised_response or ""),
      res13.revised_response)
check("c13 grounded kept",
      FX_ORDER["response"].split(".")[0][:25].lower()
      in (res13.revised_response or "").lower(), res13.revised_response)

print("── case 14: ungrounded candidate, two candidates → retry path ──")
res14 = audit(FX_ORDER["instruction"],
              a1(FX_ORDER["instruction"], FX_ORDER["intent"]),
              PLAN,
              a2("Completely unrelated text goes nowhere near this intent "
                 "at all and says nothing useful.",
                 evidence_of(FX_ORDER) + evidence_of(FX_POLICY)))
check("c14 verdict flag", res14.verdict == "flag", res14.to_dict())
check("c14 retry offered", res14.retry_candidate_rank == 2,
      res14.retry_candidate_rank)

print("── case 15: full audit_with_retry loop ──")
final, retried = audit_with_retry(
    FX_ORDER["instruction"],
    a1(FX_ORDER["instruction"], FX_ORDER["intent"]),
    PLAN,
    a2(FX_POLICY["response"],
       evidence_of(FX_POLICY) + evidence_of(FX_ORDER)))
check("c15 retry executed", retried is True, retried)
check("c15 final verdict accept", final.verdict == "accept", final.to_dict())
check("c15 retry annotation", any(i.startswith("retry:") for i in final.issues),
      final.issues)

# ── Property test over 500 sampled TRAIN responses ──────────────────────
print("\n── property test: 500 sampled TRAIN responses ──")
emb = None
for c in glob.glob("outputs/cache/train*BAAI*.npy"):
    arr = np.load(c)
    if arr.shape == (len(train), 384):
        emb = arr
        break
assert emb is not None, "TRAIN embedding cache missing"
emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

random.seed(0)
sample = random.sample(range(len(train)), 500)
tally = {"accept": 0, "revise": 0, "flag": 0}
violations: list[str] = []

for i in sample:
    row = train.iloc[i]
    intent = str(row.intent)
    qtext = str(row.instruction)
    response = str(row.response)

    # evidence: self + top-4 same-intent neighbours (simulates A2 retrieval)
    m = np.where(train["intent"].values == intent)[0]
    m = m[m != i]
    order = np.argsort(-(emb[m] @ emb[i]))[:4]
    evid = [(qtext, response, intent)]
    for j in order:
        evid.append((str(train.iloc[j].instruction),
                     str(train.iloc[j].response), intent))

    res = audit(qtext, a1(qtext, intent), PLAN, a2(response, evid))
    tally[res.verdict] += 1

    text = response if res.verdict == "accept" else (res.revised_response or "")
    stripped = strip_placeholders(text)
    if claim_matches(stripped):
        violations.append(f"row {i} ({res.verdict}): claim survives: "
                          f"{[mm.group(0) for mm in claim_matches(stripped)]!r} "
                          f"in {text[:120]!r}")
    if res.verdict in ("accept", "revise") and "{{" in text:
        violations.append(f"row {i} ({res.verdict}): unresolved "
                          f"placeholder in {text[:120]!r}")

summary = (f"verdicts on 500 TRAIN responses: accepted={tally['accept']} "
           f"revised={tally['revise']} flagged={tally['flag']}")
print(summary)
check("property: no claim / placeholder in accepted or revised outputs",
      not violations, violations[:5])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL PASS")
print(summary)
