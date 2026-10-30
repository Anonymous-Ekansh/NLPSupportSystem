"""Agent A3 — Routing / Planning.

Decides the next step from the query, A1's classification and the available
evidence, and explains why. Deterministic, no LLM calls, throttled only by the
policy file.

Rule order (first match wins; earlier decisions are deliberate)
---------------------------------------------------------------
1. empty / gibberish                      -> escalate
2. sensitive content (legal threat, fraud, chargeback, harassment, self-harm)
                                          -> escalate
3. out-of-distribution (top1_similarity)  -> escalate
4. required detail missing for the intent -> clarify (names the missing slots)
5. low confidence                         -> escalate
6. ambiguous top-2 (small margin / mid confidence)
      same category                       -> answer, widened retrieval
      different category                  -> clarify between the two intents
7. otherwise                              -> answer, standard retrieval

Why OOD sits just above slot detection but below sensitive content: a clear
out-of-domain query ("what is the weather in Paris") can be classified with high
confidence yet has a low neighbour similarity, and asking it for an order number
would be nonsense. The required case "I need a refund but cannot find my order
number" is not OOD (its similarity sits above the 0.55 floor), so it still reaches
the clarify path with the missing slot named. Sensitive content stays at the top:
a legal threat must reach a human even if the user also forgot an order number.

Thresholds never live here: confidence bands, top-2 margin, OOD limit,
retrieval k and the sensitive-term lists all come from `config/l2_policy.json`
(see `nlp/l2/policy.py`) and are tuned on DEV by `scripts/tune_a3_policy.py`.
"""

import re
from typing import Optional

from nlp.l2.contracts import (
    ClarifyOption,
    MissingSlot,
    RetrievalSettings,
    RoutePlan,
)
from nlp.l2.policy import get as pget
from nlp.l2.policy import load_policy
from nlp.l2.slots import has_absence_phrase, missing_slots

_VOWELS = set("aeiouy")


# ── Gibberish / emptiness ───────────────────────────────────────────────

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def is_empty(query: str) -> bool:
    return not (query or "").strip()


def is_gibberish(query: str, policy: Optional[dict] = None) -> bool:
    """Cheap, deterministic junk detector (no vocabulary model needed).

    Flags: no letters/digits at all; extremely long single tokens (mash);
    tokens with no vowel at all; very low vowel ratio over a long letter run;
    and repeated-character runs ("aaaaaa"). It deliberately does not try to
    judge grammar or meaning — that is A1's job.
    """
    policy = policy or load_policy()
    q = (query or "").strip()
    if not q:
        return False  # emptiness is its own route, not gibberish

    if not re.search(r"[a-z0-9]", q.lower()):
        return True

    min_chars = int(pget(policy, "routing.gibberish_min_chars", 3))
    max_token = int(pget(policy, "routing.gibberish_max_token_chars", 25))

    toks = _tokens(q)
    if not toks:
        return True
    if any(len(t) > max_token for t in toks):
        return True
    words = [t for t in toks if t.isalpha()]
    # repeated-character runs must be all-letter (order numbers like 37777
    # are not gibberish)
    if any(len(w) >= 5 and len(set(w)) <= 2 for w in words):
        return True

    no_vowel_words = [w for w in words if len(w) >= 3 and not (_VOWELS & set(w))]
    if words and len(no_vowel_words) == len(words):
        return True
    if words:
        letters = "".join(words)
        if len(letters) >= 8:
            vowel_ratio = sum(ch in _VOWELS for ch in letters) / len(letters)
            if vowel_ratio < 0.15:
                return True

    # Short inputs with no alphabetic word at all (e.g. "12345", "!!!")
    if len(q) < min_chars and not words:
        return True
    return False


# ── Sensitive content ───────────────────────────────────────────────────

def find_sensitive_terms(query: str, policy: Optional[dict] = None) -> list[str]:
    """Return `["group: term", ...]` for every sensitive phrase present."""
    policy = policy or load_policy()
    lowered = (query or "").lower()
    hits: list[str] = []
    for group, terms in (pget(policy, "escalation.sensitive_terms", {}) or {}).items():
        for term in terms:
            if term in lowered:
                hits.append(f"{group}: {term}")
    return hits


# ── Clarify helpers ─────────────────────────────────────────────────────

_INFORMATIONAL_RE = re.compile(
    r"^(?:how (?:can|do|does|would|should) (?:i|we|you)\b|how to\b|"
    r"is it possible to\b|"
    r"can (?:i|we) (?:get|request|receive|have|check|find|track|see|cancel|"
    r"change|claim)\b|"
    r"what (?:is|are) the\b)",
    re.IGNORECASE,
)


def is_informational(query: str) -> bool:
    """How-to / eligibility phrasing asks for the procedure, not a value.

    "How can I request a refund?" must be answered with the process even
    though `get_refund` nominally requires an order number: explaining a
    procedure cannot claim that any action happened.  An explicit absence
    phrase ("I cannot find my order number") is checked separately by the
    caller and still forces the clarify path — that is the Level-2 contract
    from the problem statement.
    """
    return bool(_INFORMATIONAL_RE.match((query or "").strip()))


def _intent_question(intent: str, category: str = "") -> str:
    label = intent.replace("_", " ")
    if category:
        return f"Do you need help with {label} (about your {category.lower()})?"
    return f"Do you need help with {label}?"


def _clarify_options(a1_output: dict, n: int = 2) -> list[ClarifyOption]:
    options = []
    for item in (a1_output.get("top3") or [])[:n]:
        intent = item.get("intent", "")
        options.append(ClarifyOption(
            intent=intent,
            prob=round(float(item.get("prob", 0.0)), 4),
            question=_intent_question(intent),
            category=a1_output.get("category", "") if intent == a1_output.get("intent") else "",
        ))
    return options


def _describe_missing(slots: list[MissingSlot]) -> str:
    labels = [s.label for s in slots]
    if len(labels) == 1:
        return f"the {labels[0]}"
    return ", ".join(labels[:-1]) + " and " + labels[-1]


# ── Planner ─────────────────────────────────────────────────────────────

class RoutePlanner:
    """Agent A3. `plan(query, a1_output) -> RoutePlan`."""

    name = "RoutePlanner"
    role = "routing / planning"

    def __init__(self, policy: Optional[dict] = None, policy_path: Optional[str] = None):
        self._policy = policy
        self._policy_path = policy_path

    @property
    def policy(self) -> dict:
        if self._policy is not None:
            return self._policy
        return load_policy(self._policy_path)

    # ── entry point ───────────────────────────────────────────────────

    def plan(self, query: str, a1_output: Optional[dict] = None) -> RoutePlan:
        """Select the next step for *query*. Always returns a plan."""
        policy = self.policy
        a1_output = a1_output or {}

        intent = str(a1_output.get("intent") or "")
        category = str(a1_output.get("category") or "")
        confidence = float(a1_output.get("confidence") or 0.0)
        margin = float(a1_output.get("margin") or 0.0)
        similarity = float(a1_output.get("top1_similarity") or 0.0)
        top3 = a1_output.get("top3") or []
        top2 = [t.get("intent", "") for t in top3[:2]]

        conf_high = float(pget(policy, "thresholds.confidence_high", 0.75))
        conf_low = float(pget(policy, "thresholds.confidence_low", 0.40))
        margin_min = float(pget(policy, "thresholds.top2_margin_min", 0.20))
        ood_min = float(pget(policy, "thresholds.ood_similarity_min", 0.60))
        k = int(pget(policy, "retrieval.k", 5))
        k_widened = int(pget(policy, "retrieval.k_widened", 10))
        mode = str(pget(policy, "retrieval.mode", "hybrid"))

        reasons: list[str] = []
        codes: list[str] = []

        def plan_route(route: str, retrieval: RetrievalSettings,
                       clarify_options: Optional[list[ClarifyOption]] = None,
                       missing: Optional[list[MissingSlot]] = None) -> RoutePlan:
            return RoutePlan(
                route=route,
                reasons=reasons,
                reason_codes=codes,
                retrieval=retrieval,
                missing_slots=missing or [],
                clarify_options=clarify_options or [],
                intent=intent,
                category=category,
                confidence=round(confidence, 4),
            )

        # 1. empty / junk ────────────────────────────────────────────────
        if is_empty(query):
            reasons.append("The query is empty, so there is nothing to answer.")
            codes.append("empty_query")
            return plan_route("escalate", RetrievalSettings(mode=mode, k=k))

        if is_gibberish(query, policy):
            reasons.append(
                "The query does not look like natural language, so no intent "
                "can be trusted.")
            codes.append("gibberish")
            return plan_route("escalate", RetrievalSettings(mode=mode, k=k))

        # 2. sensitive content ───────────────────────────────────────────
        sensitive = find_sensitive_terms(query, policy)
        if sensitive:
            groups = sorted({hit.split(":", 1)[0] for hit in sensitive})
            reasons.append(
                "The query mentions sensitive content ("
                + ", ".join(hits.replace("_", " ") for hits in sensitive)
                + "), which must go to a human agent.")
            codes.append("sensitive_content")
            codes.extend(f"sensitive_{g}" for g in groups)
            return plan_route("escalate", RetrievalSettings(mode=mode, k=k))

        # 3. out of distribution ─────────────────────────────────────────
        if similarity and similarity < ood_min:
            reasons.append(
                f"The query is not close to anything in the known catalogue "
                f"(top-1 similarity {similarity:.2f} < {ood_min:.2f}), so it is "
                f"treated as out of scope.")
            codes.append("out_of_distribution")
            return plan_route("escalate", RetrievalSettings(mode=mode, k=k))

        # 4. missing required information ────────────────────────────────
        if pget(policy, "routing.clarify_on_missing_slot", True) and intent:
            missing = missing_slots(query, intent, policy)
            if (missing and is_informational(query)
                    and not has_absence_phrase(query, policy)):
                # How-to/eligibility question: explain the procedure, do not
                # demand a value the user never claimed to have.
                reasons.append(
                    f"The query asks how to handle '{intent}' rather than "
                    f"requesting it now, so it can be answered with the "
                    f"procedure without {_describe_missing(missing)}; "
                    f"explaining a process claims no action.")
                codes.append("informational_question")
            elif missing:
                reasons.append(
                    f"The intent '{intent}' needs {_describe_missing(missing)} "
                    f"before an answer can be given; asking for it prevents a "
                    f"wrong answer (e.g. 'refund processed' with no order).")
                codes.append("missing_information")
                codes.extend(f"missing_{m.slot}" for m in missing)
                absence = has_absence_phrase(query, policy)
                if absence:
                    reasons.append(
                        f"The user states they cannot supply it ('{absence}'), "
                        f"so the reply must request it explicitly rather than "
                        f"assume a value.")
                    codes.append("user_lacks_slot")
                # Very short / underspecified queries also carry the two
                # competing intents, so A5 can ask "X or Y?" alongside the
                # missing item (a bare "refund" is ambiguous in practice).
                options = (_clarify_options(a1_output, 2)
                           if len(_tokens(query)) <= 4 else None)
                return plan_route(
                    "clarify",
                    RetrievalSettings(mode=mode, k=k,
                                      intents=[intent] if intent else []),
                    clarify_options=options,
                    missing=missing,
                )

        # 5. low confidence ──────────────────────────────────────────────
        if confidence < conf_low:
            reasons.append(
                f"Confidence {confidence:.2f} is below the escalation floor "
                f"{conf_low:.2f}; a wrong automatic answer is worse than a "
                f"human handoff.")
            codes.append("low_confidence")
            return plan_route("escalate", RetrievalSettings(mode=mode, k=k))

        # 6. ambiguity ───────────────────────────────────────────────────
        ambiguous = margin < margin_min or confidence < conf_high
        if ambiguous and len(top2) == 2:
            cat_a = _category_of(top2[0], a1_output, policy)
            cat_b = _category_of(top2[1], a1_output, policy)
            same_category = bool(cat_a) and cat_a == cat_b

            if (same_category and pget(policy, "routing.answer_when_top2_same_category", True)):
                reasons.append(
                    f"Top-2 intents '{top2[0]}' and '{top2[1]}' are close "
                    f"(confidence {confidence:.2f}, margin {margin:.2f}) but "
                    f"share a category, so retrieval is widened instead of "
                    f"interrupting the user.")
                codes.extend(["ambiguous_same_category", "widened_retrieval"])
                return plan_route(
                    "answer",
                    RetrievalSettings(mode=mode, k=k_widened, intents=top2,
                                      widen=True),
                )

            options = _clarify_options(a1_output, 2)
            reasons.append(
                f"Two intents are almost equally likely ('{top2[0]}' and "
                f"'{top2[1]}', margin {margin:.2f} < {margin_min:.2f}), so the "
                f"cheapest correct move is to ask which one is meant.")
            codes.append("ambiguous_intent")
            return plan_route(
                "clarify",
                RetrievalSettings(mode=mode, k=k, intents=top2),
                clarify_options=options,
            )

        # 7. answer ──────────────────────────────────────────────────────
        reasons.append(
            f"Intent '{intent}' is predicted with confidence {confidence:.2f} "
            f"(margin {margin:.2f}), above the clarify floor {conf_low:.2f}, and "
            f"no required detail is missing, so retrieval can answer directly.")
        codes.append("standard_retrieval")
        return plan_route(
            "answer",
            RetrievalSettings(mode=mode, k=k, intents=[intent] if intent else []),
        )

    # convenience alias used by the pipeline wrapper
    def run(self, query: str, a1_output: Optional[dict] = None) -> RoutePlan:
        return self.plan(query, a1_output)


def _category_of(intent: str, a1_output: dict, policy: dict) -> str:
    """Category for a top-k intent.

    A1 only returns the category of its top-1 intent, so the other candidates
    are resolved through the static intent->category map in the policy file
    (`intents`), which was derived from TRAIN majority votes.
    """
    if intent == a1_output.get("intent"):
        return str(a1_output.get("category") or "")
    return str(pget(policy, "intents", {}).get(intent) or "")


def route_to_decision(route: str) -> str:
    """Map an A3 route to the A5 outcome vocabulary."""
    return {"answer": "ANSWER", "clarify": "CLARIFY", "escalate": "ESCALATE"}[route]


__all__ = [
    "RoutePlanner", "is_empty", "is_gibberish", "find_sensitive_terms",
    "is_informational", "route_to_decision",
]
