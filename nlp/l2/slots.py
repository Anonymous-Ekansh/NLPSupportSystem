"""Slot detection: does the user's query contain the information the intent needs?

The problem statement is explicit: "I need a refund but cannot find my order
number" must lead to a request for details, never to "refund processed".
This module is the single place that answers "what is missing?".

How the required-slot map is built
----------------------------------
1. Data evidence, TRAIN only: `derive_required_slots()` reads
   `outputs/splits/train.csv` and counts the `{{placeholder}}` names used by
   each intent (instructions = user-side slots, responses = assistant-side
   slots). The ratios are recorded under `slots.derived_from_train` in
   `config/l2_policy.json` so the provenance of every requirement is visible.
2. Policy choice: `slots.required` in the same file is what the runtime uses.
   It is seeded from (1) plus explicit overrides, because the data alone does
   not demand an order number for `get_refund` (only 12 % of TRAIN get_refund
   responses carry a refund amount) while the product requirement does.

Detectors only look for *usable values* (a number, an email, an amount, a
date), never for the word "order number" alone — mentioning the slot without a
value is exactly the case that must trigger a clarifying question.

No LLM calls, no new dependencies: regexes + phrase lists from the policy.
"""

import re
from typing import Optional

from nlp.l2.contracts import MissingSlot
from nlp.l2.policy import load_policy

# ── Value detectors ─────────────────────────────────────────────────────
# Each pattern must capture a *value*. Matching the bare slot name is a bug.
_ORDERISH = r"(?:order|purchase)"
_ORDER_NUM = re.compile(
    rf"\b{_ORDERISH}(?:\s*(?:number|no\.?|id|code|ref(?:erence)?))?"
    r"\s*(?:is|was|:|-|#)?\s*([A-Z]{0,4}[-_#]?\d{2,})\b",
    re.IGNORECASE,
)
_HASH_NUM = re.compile(r"#\s?(\d{3,})\b")
_BARE_LONG_NUM = re.compile(r"\b(\d{6,})\b")

_INVOICE_NUM = re.compile(
    r"\binvoice(?:\s*(?:number|no\.?|id|ref(?:erence)?))?"
    r"\s*(?:is|:|-|#)?\s*([A-Z]{0,4}[-_#]?\d{2,})\b",
    re.IGNORECASE,
)
_INV_PREFIXED = re.compile(r"\b(INV[-_]?\d{2,})\b", re.IGNORECASE)

_TRACKING_NUM = re.compile(
    r"\b(?:tracking|track(?:ing)?\s*number|consignment|awb)"
    r"\s*(?:is|:|-|#)?\s*([A-Z0-9]{6,})\b"
)
_TRACKING_RAW = re.compile(r"\b(1Z[0-9A-Z]{10,})\b")

_EMAIL = re.compile(r"\b([\w.+-]+@[\w-]+\.[\w.-]+)\b")

_AMOUNT = re.compile(
    r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d{1,2})?)"
    r"|\b(\d[\d,]*(?:\.\d{1,2})?\s?(?:dollars?|usd|eur|euros?|gbp|pounds?"
    r"|inr|rupees?|bucks))\b",
    re.IGNORECASE,
)

_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"(?:,?\s*\d{4})?|yesterday|today|last\s+(?:week|month|year|monday|tuesday"
    r"|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)

_PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")

DETECTORS = {
    "order_number": (_ORDER_NUM, _HASH_NUM, _BARE_LONG_NUM),
    "invoice_number": (_INVOICE_NUM, _INV_PREFIXED),
    "tracking_number": (_TRACKING_NUM, _TRACKING_RAW),
    "email": (_EMAIL,),
    "amount": (_AMOUNT,),
    "date": (_DATE,),
    "phone": (_PHONE,),
}

# Slot-name mentions, used for evidence text ("user mentions the order number
# but gives no value").
_MENTIONS = {
    "order_number": re.compile(r"\b(order|purchase)\b(?:\s*(?:number|no\.?|id))?",
                               re.IGNORECASE),
    "invoice_number": re.compile(r"\binvoice\b(?:\s*(?:number|no\.?|id))?", re.IGNORECASE),
    "tracking_number": re.compile(r"\b(tracking|consignment|awb)\b", re.IGNORECASE),
    "email": re.compile(r"\b(e-?mail)\b", re.IGNORECASE),
    "amount": re.compile(r"\b(amount|refund|paid|charged|money)\b", re.IGNORECASE),
    "date": re.compile(r"\b(date|when|day)\b", re.IGNORECASE),
    "phone": re.compile(r"\b(phone|mobile|number to call)\b", re.IGNORECASE),
}

# Placeholder name -> slot id (for derive_required_slots provenance).
PLACEHOLDER_TO_SLOT = {
    "Order Number": "order_number",
    "Order/Transaction ID": "order_number",
    "Order/Claim Number": "order_number",
    "Invoice Number": "invoice_number",
    "Tracking Number": "tracking_number",
    "Rebate Tracking Number": "tracking_number",
    "Customer Support Email": "email",
    "Customer Support Email Address": "email",
    "Refund Amount": "amount",
    "Money Amount": "amount",
    "Currency Symbol": "amount",
    "Date": "date",
    "Date Range": "date",
}

# Reverse map: slot -> placeholder names. The benchmark's instructions carry
# the *literal* text "{{Order Number}}" where a real user would type a value,
# so accepting that literal as a supplied value is required to score the
# dataset honestly (real users produce values, benchmark rows produce tokens).
SLOT_PLACEHOLDERS: dict[str, set[str]] = {}
for _name, _slot in PLACEHOLDER_TO_SLOT.items():
    SLOT_PLACEHOLDERS.setdefault(_slot, set()).add(_name.lower())

_PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}")


# ── Low-level helpers ───────────────────────────────────────────────────

def find_value(text: str, slot: str) -> Optional[str]:
    """Return the matched value string for *slot*, or None.

    Accepts both real values (digits, emails, amounts, dates) and the
    benchmark's `{{Order Number}}`-style placeholder literal, which stands in
    for a user-supplied value in the dataset.
    """
    for pattern in DETECTORS.get(slot, ()):
        m = pattern.search(text or "")
        if m:
            return (m.group(1) if m.groups() else m.group(0)).strip()
    found_placeholders = _PLACEHOLDER_RE.findall(text or "")
    literals = SLOT_PLACEHOLDERS.get(slot, ())
    for name in found_placeholders:
        if name.strip().lower() in literals:
            return f"{{{{{name.strip()}}}}}"
    return None


def mentions_slot(text: str, slot: str) -> bool:
    pattern = _MENTIONS.get(slot)
    return bool(pattern and pattern.search(text or ""))


def has_absence_phrase(text: str, policy: Optional[dict] = None) -> str:
    """Return the matching 'I don't have it' phrase, or ''."""
    policy = policy or load_policy()
    lowered = (text or "").lower()
    for phrase in policy["slots"].get("not_have_phrases", []):
        if phrase in lowered:
            return phrase
    return ""


def detect_slots(text: str, slots: Optional[list[str]] = None) -> dict[str, str]:
    """Map every requested slot to its found value (absent slots are omitted)."""
    names = slots if slots is not None else list(DETECTORS)
    found = {}
    for slot in names:
        value = find_value(text, slot)
        if value:
            found[slot] = value
    return found


# ── Main entry point ────────────────────────────────────────────────────

def missing_slots(
    query: str,
    intent: str,
    policy: Optional[dict] = None,
) -> list[MissingSlot]:
    """Return the required slots of *intent* that *query* does not supply.

    A slot is missing when no usable value is found. If the user explicitly
    says they do not have it ("can't find", "lost", "don't have"), that phrase
    is kept as evidence so the trace/report can show why we asked.
    """
    policy = policy or load_policy()
    required = policy["slots"].get("required", {}).get(intent, [])
    if not required:
        return []

    absence = has_absence_phrase(query, policy)
    labels = policy["slots"].get("slot_labels", {})
    questions = policy["slots"].get("questions", {})

    out: list[MissingSlot] = []
    for slot in required:
        if find_value(query, slot):
            continue
        label = labels.get(slot, slot.replace("_", " "))
        if absence and mentions_slot(query, slot):
            evidence = (f"user says '{absence}' about the {label} and supplies "
                        f"no value")
        elif absence:
            evidence = f"no {label} value in query (user says '{absence}')"
        else:
            evidence = f"no {label} value in query"
        out.append(MissingSlot(
            slot=slot,
            label=label,
            question=questions.get(slot, f"Could you share your {label}?"),
            evidence=evidence,
        ))
    return out


# ── TRAIN-only derivation (provenance / regeneration) ───────────────────

def derive_required_slots(
    train_csv: str = "outputs/splits/train.csv",
    min_ratio: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Count placeholder usage per intent on TRAIN only.

    Returns `{intent: {slot: ratio}}` for placeholder names with a known slot
    mapping. Ratios are computed over `max(n_instructions, n_responses)` so an
    assistant-side-only placeholder (e.g. delivery_period's tracking number)
    still surfaces, but a value is only kept when it crosses `min_ratio`.

    This is a *reporting* helper; the runtime reads `slots.required` from the
    policy file. pandas is imported here (not at module import) so importing
    `nlp.l2.slots` stays dependency-light.
    """
    import pandas as pd

    df = pd.read_csv(train_csv)
    counts: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for row in df.itertuples(index=False):
        intent = row.intent
        totals[intent] = totals.get(intent, 0) + 1
        places = set()
        for text in (str(row.instruction), str(row.response)):
            for name in _PLACEHOLDER_RE.findall(text):
                slot = PLACEHOLDER_TO_SLOT.get(name.strip())
                if slot:
                    places.add(slot)
        for slot in places:
            counts.setdefault(intent, {}).setdefault(slot, 0)
            counts[intent][slot] += 1

    derived: dict[str, dict[str, float]] = {}
    for intent, slots in counts.items():
        total = totals[intent]
        kept = {s: round(c / total, 2) for s, c in slots.items()
                if c / total >= min_ratio}
        if kept:
            derived[intent] = dict(sorted(kept.items(), key=lambda kv: -kv[1]))
    return derived


__all__ = [
    "DETECTORS", "PLACEHOLDER_TO_SLOT",
    "find_value", "mentions_slot", "has_absence_phrase",
    "detect_slots", "missing_slots", "derive_required_slots",
]
