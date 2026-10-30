"""Agent A5 — Coordinator: the one decision, and the full agent trace.

`coordinate(query, a1, plan, a2, audit, retry_info)` turns the outputs of
A1 (intent), A3 (route plan), A2 (candidate response) and A4 (audit) into
exactly one of ANSWER / CLARIFY / ESCALATE, plus a five-step trace
(A1, A3, A2, A4, A5) where every step carries role, input, output,
decision, evidence and latency.

Decision rules (first match wins)
----------------------------------
1. plan.route == "clarify"   -> CLARIFY.  Missing slot: one focused
   follow-up built from the slot's question (plus what to do when the user
   says they cannot find it).  Ambiguity: "Are you asking about X or Y?"
   from `plan.clarify_options` using human-readable intent names.
2. plan.route == "escalate"  -> ESCALATE with a human-handoff summary:
   reason codes, detected intent, evidence and a suggested queue.  Never
   claims any action happened; it explicitly states nothing was changed.
3. plan.route == "answer"    -> ANSWER only when A4 passed the candidate:
   accept -> A2's response verbatim; revise -> A4's rewritten response.
4. A4 still flags after the pipeline's one retry:
     contradiction -> ESCALATE, listing the conflicting evidence;
     intent alignment -> CLARIFY between the competing intents;
     grounding / empty / unfixable claims -> ESCALATE (unsupported output).

No LLM calls.  Every threshold/template is local text; the safety-critical
wording (no completion claims) is enforced by construction and by A4.
"""

from __future__ import annotations

from typing import Optional

from nlp.l2.contracts import (
    AuditResult,
    FinalDecision,
    RoutePlan,
    TraceStep,
)
from nlp.l2.policy import get as pget
from nlp.l2.policy import load_policy

__all__ = ["coordinate", "intent_label", "suggest_queue"]

# ── Human-readable intent names (exact dataset labels, no new labels) ────

_INTENT_LABELS = {
    "cancel_order": "cancelling an order",
    "change_order": "changing an order",
    "change_shipping_address": "changing your shipping address",
    "check_cancellation_fee": "cancellation fees",
    "check_invoice": "checking an invoice",
    "check_payment_methods": "payment methods",
    "check_refund_policy": "the refund policy",
    "complaint": "filing a complaint",
    "contact_customer_service": "contacting customer service",
    "contact_human_agent": "talking to a human agent",
    "create_account": "creating an account",
    "delete_account": "deleting your account",
    "delivery_options": "delivery options",
    "delivery_period": "delivery timeframes",
    "edit_account": "editing your account",
    "get_invoice": "receiving an invoice",
    "get_refund": "requesting a refund",
    "newsletter_subscription": "newsletter subscriptions",
    "payment_issue": "a payment issue",
    "place_order": "placing an order",
    "recover_password": "password recovery",
    "registration_problems": "registration problems",
    "review": "leaving a review",
    "set_up_shipping_address": "setting up a shipping address",
    "switch_account": "switching accounts",
    "track_order": "tracking your order",
    "track_refund": "tracking your refund",
}


def intent_label(intent: str) -> str:
    """Human-readable phrasing for an exact dataset intent label."""
    if not intent:
        return "your request"
    return _INTENT_LABELS.get(intent, intent.replace("_", " "))


# ── Suggested human queue ────────────────────────────────────────────────

_SENSITIVE_QUEUES = {
    "legal_threat": "legal-escalations",
    "fraud": "fraud-review",
    "chargeback": "payments-dispute",
    "harassment": "trust-and-safety",
    "self_harm": "crisis-support",
}

_CATEGORY_QUEUES = {
    "ACCOUNT": "account-support",
    "CANCEL": "cancellations",
    "CONTACT": "general-support",
    "DELIVERY": "delivery",
    "FEEDBACK": "feedback",
    "INVOICE": "invoicing",
    "ORDER": "order-support",
    "PAYMENT": "payments",
    "REFUND": "refunds",
    "SHIPPING": "shipping",
    "SUBSCRIPTION": "subscriptions",
}

_GENERIC_CODES = {"gibberish", "empty_query", "out_of_distribution"}


def suggest_queue(reason_codes: list, category: str,
                  failed_checks: Optional[list] = None) -> str:
    """Map reason codes / failed checks to a human queue name."""
    for code in reason_codes:
        if code.startswith("sensitive_"):
            group = code[len("sensitive_"):]
            if group in _SENSITIVE_QUEUES:
                return _SENSITIVE_QUEUES[group]
    failed = set(failed_checks or [])
    if "contradiction" in failed:
        return "quality-review"
    if set(reason_codes) & _GENERIC_CODES:
        return "general-support"
    if "audit_flag" in reason_codes or "unsupported_output" in reason_codes:
        return "quality-review"
    return _CATEGORY_QUEUES.get(category, "general-support")


# ── CLARIFY text ─────────────────────────────────────────────────────────

def _options_question(plan: RoutePlan) -> str:
    opts = plan.clarify_options
    labels = [intent_label(o.intent) for o in opts[:2]]
    if len(labels) == 2:
        return f"Are you asking about {labels[0]} or {labels[1]}?"
    if len(labels) == 1:
        return f"Just to be sure — are you asking about {labels[0]}?"
    return ""


def _missing_question(plan: RoutePlan, reason_codes: list) -> str:
    slots = plan.missing_slots
    if not slots:
        return ""
    primary = slots[0]
    parts = [primary.question]
    if len(slots) > 1:
        rest = ", ".join(s.label for s in slots[1:])
        parts.append(f"It would also help to know the {rest}.")
    if "user_lacks_slot" in reason_codes and primary.slot == "order_number":
        parts.append(
            "If you cannot find your order number, reply with the email "
            "address you used to place the order instead.")
    return " ".join(parts)


def _clarify_response(plan: RoutePlan, a1: dict, reason_codes: list) -> str:
    """One focused follow-up: the missing item, or the competing intents."""
    missing = _missing_question(plan, reason_codes)
    options = _options_question(plan)
    if missing and options:
        return f"{missing} {options}"
    if missing:
        return missing
    if options:
        return options
    label = intent_label(str(a1.get("intent", "")))
    return (f"Could you tell me a little more about what you need with "
            f"{label}?")


# ── ESCALATE text ────────────────────────────────────────────────────────

def _escalate_response(a1: dict, plan: RoutePlan, reason_codes: list,
                       evidence_lines: list, queue: str) -> str:
    """Human-handoff summary. States explicitly that nothing was changed."""
    intent = str(a1.get("intent", ""))
    category = str(a1.get("category", ""))
    topic = intent_label(intent)
    parts = [
        "I'm passing this to a human specialist for review — I don't want "
        "to guess and get it wrong. No account, order, payment or refund "
        "changes have been made.",
        f"Detected intent: {topic}"
        + (f" ({intent}" + (f", {category}" if category else "") + ")"
           if intent else "") + ".",
        "Reasons: " + ", ".join(reason_codes) + ".",
    ]
    if evidence_lines:
        parts.append("Evidence: " + " | ".join(evidence_lines[:4]))
    parts.append(f"Suggested queue: {queue}.")
    return " ".join(parts)


# ── Trace helpers ────────────────────────────────────────────────────────

def _step(agent: str, role: str, decision: str, evidence: list,
          latency_ms: float, input_obj: dict, output_obj: dict,
          output_summary: str) -> TraceStep:
    return TraceStep(
        agent=agent,
        role=role,
        input_summary=str(input_obj.get("query", ""))[:160],
        output_summary=output_summary[:400],
        decision=decision,
        evidence=[str(e) for e in evidence],
        latency_ms=round(float(latency_ms), 1),
        payload={"input": input_obj, "output": output_obj},
    )


def _build_trace(query: str, a1: dict, plan: RoutePlan,
                 a2: Optional[dict], audit: Optional[AuditResult],
                 retry_info: dict, final: FinalDecision) -> list:
    """Five steps: A1, A3, A2, A4, A5 — always present, even if skipped."""
    lat = retry_info.get("latency_ms") or {}
    intent = str(a1.get("intent", ""))

    steps = [
        _step("A1", "intent classification", intent,
              [intent] if intent else [], lat.get("a1", 0.0),
              {"query": query}, a1,
              f"{intent} (conf {a1.get('confidence', 0)}, "
              f"sim {a1.get('top1_similarity', 0)})"),
        _step("A3", "routing / planning", plan.route.upper(),
              list(plan.reason_codes), lat.get("a3", 0.0),
              {"query": query, "a1_output": {"intent": intent,
                                             "confidence": a1.get("confidence")}},
              plan.to_dict(),
              f"route={plan.route}; " + "; ".join(plan.reasons[:2])),
    ]

    # A2 — skipped on clarify/escalate routes (nothing to retrieve for)
    if a2 is None:
        steps.append(_step(
            "A2", "response retrieval", "skipped", [], lat.get("a2", 0.0),
            {"query": query}, {"skipped": True, "reason": f"a3_route_{plan.route}"},
            f"skipped: A3 chose {plan.route}, no candidate needed"))
    else:
        ev_ids = [f"ex{i}" for i in range(len(a2.get("supporting_examples") or []))]
        steps.append(_step(
            "A2", "response retrieval", "candidate_produced", ev_ids,
            lat.get("a2", 0.0), {"query": query}, a2,
            f"retrieval_score={a2.get('retrieval_score')} "
            f"self_check_passed={bool((a2.get('self_check') or {}).get('check_passed'))}"))

    # A4 — audit (with one optional retry recorded inside the step)
    if audit is None:
        steps.append(_step(
            "A4", "response auditing", "skipped", [], lat.get("a4", 0.0),
            {"query": query}, {"skipped": True,
                               "reason": f"a3_route_{plan.route}"},
            f"skipped: A3 chose {plan.route}"))
    else:
        payload = audit.to_dict()
        if retry_info.get("retried"):
            first = retry_info.get("first_audit")
            payload["retry"] = {
                "attempted": True,
                "first_verdict": first.verdict if first is not None else None,
                "first_issues": list(first.issues) if first is not None else [],
                "a2_retried": retry_info.get("a2_first") is not None,
            }
        steps.append(_step(
            "A4", "response auditing", audit.verdict.upper(),
            list(audit.evidence_ids),
            lat.get("a4", 0.0) + lat.get("a4_retry", 0.0),
            {"query": query}, payload,
            f"verdict={audit.verdict}; failed={audit.failed_checks}"))

    steps.append(_step(
        "A5", "coordination / final decision", final.decision,
        list(final.evidence_ids), lat.get("a5", 0.0),
        {"query": query, "plan_route": plan.route,
         "audit_verdict": audit.verdict if audit else None},
        final.to_dict(),
        f"{final.decision}: {final.response[:120]}"))
    return steps


# ── The coordinator ──────────────────────────────────────────────────────

def coordinate(
    query: str,
    a1: dict,
    plan: RoutePlan,
    a2: Optional[dict],
    audit: Optional[AuditResult],
    retry_info: Optional[dict] = None,
) -> tuple[FinalDecision, list[TraceStep]]:
    """Return the single final decision and the full five-agent trace."""
    retry_info = retry_info or {}
    policy = load_policy()
    reason_codes = list(plan.reason_codes)
    notes = list(plan.reasons)
    failed_checks: list = list(audit.failed_checks) if audit else []

    # 1/2 — A3 already decided clarify or escalate: honour it.
    if plan.route == "clarify":
        response = _clarify_response(plan, a1, reason_codes)
        decision = FinalDecision(
            decision="CLARIFY",
            response=response,
            intent=str(a1.get("intent", "")),
            category=str(a1.get("category", "")),
            reason_codes=reason_codes or ["clarify"],
            evidence_ids=([f"missing_{m.slot}"
                           for m in plan.missing_slots]
                          or [o.intent for o in plan.clarify_options]),
            notes=notes,
        )

    elif plan.route == "escalate":
        evidence_lines = list(plan.reasons)
        codes = reason_codes or ["escalate"]
        queue = suggest_queue(codes, str(a1.get("category", "")))
        response = _escalate_response(a1, plan, codes, evidence_lines, queue)
        decision = FinalDecision(
            decision="ESCALATE",
            response=response,
            intent=str(a1.get("intent", "")),
            category=str(a1.get("category", "")),
            reason_codes=codes,
            evidence_ids=[c for c in codes],
            notes=notes + [f"suggested_queue={queue}"],
        )

    # 3 — answer route: A4 must have passed the candidate.
    elif plan.route == "answer" and audit is not None and \
            audit.verdict in ("accept", "revise"):
        if audit.verdict == "accept":
            response = str((a2 or {}).get("response", ""))
            codes = reason_codes + ["audit_accept"]
        else:
            response = str(audit.revised_response or "")
            codes = reason_codes + ["audit_revise"]
        if not response.strip():
            # Defensive: nothing safe to serve -> escalate, never answer "".
            queue = suggest_queue(reason_codes + ["unsupported_output"],
                                  str(a1.get("category", "")))
            codes = reason_codes + ["unsupported_output"]
            response = _escalate_response(
                a1, plan, codes, list(audit.issues), queue)
            decision = FinalDecision(
                decision="ESCALATE", response=response,
                intent=str(a1.get("intent", "")),
                category=str(a1.get("category", "")),
                reason_codes=codes, evidence_ids=list(audit.evidence_ids),
                notes=notes + list(audit.issues),
            )
        else:
            decision = FinalDecision(
                decision="ANSWER",
                response=response,
                intent=str(a1.get("intent", "")),
                category=str(a1.get("category", "")),
                reason_codes=codes,
                evidence_ids=list(audit.evidence_ids),
                notes=notes + list(audit.issues),
            )

    # 4 — A4 still flagged (after the pipeline's one retry): map the issue.
    elif audit is not None and audit.verdict == "flag":
        codes = reason_codes + ["audit_flag"] + failed_checks
        if "contradiction" in failed_checks:
            queue = suggest_queue(codes, str(a1.get("category", "")),
                                  failed_checks)
            response = _escalate_response(
                a1, plan, codes + ["conflicting_evidence"],
                list(audit.issues), queue)
            decision = FinalDecision(
                decision="ESCALATE", response=response,
                intent=str(a1.get("intent", "")),
                category=str(a1.get("category", "")),
                reason_codes=codes + ["conflicting_evidence"],
                evidence_ids=list(audit.evidence_ids),
                notes=notes + list(audit.issues),
            )
        elif "intent_alignment" in failed_checks:
            # We are not sure what is being asked: ask, do not guess.
            codes = codes + ["uncertain_intent"]
            response = _clarify_response(plan, a1, codes)
            decision = FinalDecision(
                decision="CLARIFY", response=response,
                intent=str(a1.get("intent", "")),
                category=str(a1.get("category", "")),
                reason_codes=codes,
                evidence_ids=list(audit.evidence_ids),
                notes=notes + list(audit.issues),
            )
        else:
            # grounding / empty / unfixable claims: unsupported output
            queue = suggest_queue(codes, str(a1.get("category", "")),
                                  failed_checks)
            response = _escalate_response(
                a1, plan, codes + ["unsupported_output"],
                list(audit.issues), queue)
            decision = FinalDecision(
                decision="ESCALATE", response=response,
                intent=str(a1.get("intent", "")),
                category=str(a1.get("category", "")),
                reason_codes=codes + ["unsupported_output"],
                evidence_ids=list(audit.evidence_ids),
                notes=notes + list(audit.issues),
            )

    else:
        # answer route without an audit (or unusable states): never answer
        # an unchecked candidate.
        codes = reason_codes + ["unchecked_candidate"]
        queue = suggest_queue(codes, str(a1.get("category", "")))
        response = _escalate_response(a1, plan, codes, [], queue)
        decision = FinalDecision(
            decision="ESCALATE", response=response,
            intent=str(a1.get("intent", "")),
            category=str(a1.get("category", "")),
            reason_codes=codes, evidence_ids=[], notes=notes,
        )

    # One focused follow-up / handoff sanity: never ship an empty text.
    if not decision.response.strip():
        decision.response = (
            "I'm passing this to a human specialist for review. "
            "No account, order, payment or refund changes have been made.")

    trace = _build_trace(query, a1, plan, a2, audit, retry_info, decision)
    return decision, trace
