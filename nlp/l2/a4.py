"""Agent A4 — Response Auditor (Level 2).

Checks a candidate response (A2 output) before it can be served:

  a) intent_alignment   — evidence intents match the predicted intent and
                          query→response cosine ≥ `thresholds.similarity_min`
                          (reuses nlp.embed; no new model).
  b) grounding          — every response sentence must overlap (token
                          Jaccard ≥ `audit.grounding_jaccard_min`) with some
                          sentence of a retrieved evidence response.
  c) unsupported_claims — completion-claim lexicon/regex ("has been
                          processed", "I have cancelled", "we have issued",
                          "your refund is on its way") and invented specifics
                          (amounts / dates / long ids not present in the
                          query or the evidence).
  d) placeholder_safety — unresolved {{...}} tokens.
  e) contradiction      — retrieved neighbours disagree on intent
                          (share agreeing with the prediction below
                          `thresholds.neighbor_agreement_min`) or give
                          conflicting procedural direction (pro/con phrase
                          axes from the policy file).

Verdicts
--------
* accept   — every check passes; the response is served verbatim.
* revise   — deterministic rewrite: action claims are softened to
             conditional wording ("has been processed" → "can be
             processed"), non-fixable claim sentences and invented-specific
             sentences are dropped, and unresolved {{placeholders}} are
             replaced with neutral wording ("your order number").  The
             rewritten text must itself be free of completion claims.
* flag     — nothing safe could be produced (contradiction, alignment
             failure, or a rewrite that empties / cannot repair the text).

Retry loop: when alignment or grounding fails AND ≥2 supporting examples
exist, `retry_candidate_rank` is set to 2 so the pipeline can ask A2 for its
next-best candidate; `audit_with_retry()` re-audits that candidate once.

No LLM calls, no heavy dependencies; every threshold comes from the policy
file so `scripts/tune_a3_policy.py`-style tuning can overwrite it.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from nlp.l2.contracts import AuditResult, CheckResult, RoutePlan
from nlp.l2.policy import get, load_policy
from nlp.l2.slots import PLACEHOLDER_TO_SLOT
from nlp.normalize import normalize

__all__ = ["audit", "audit_with_retry", "build_retry_candidate",
           "tokens", "sentences", "auditable_sentences", "jaccard"]

# ── Text helpers ─────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")


def tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def sentences(text: str) -> list:
    """Split into sentence-sized chunks (whitespace already collapsed)."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in parts if s.strip()]


def auditable_sentences(text: str) -> list:
    """Sentence-level units the grounding check must account for.

    Bullet lines are kept whole; very short fragments are skipped so
    list numbering and salutations never dominate the verdict.
    """
    out: list = []
    for chunk in re.split(r"\n+", (text or "")):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.match(r"^([-*•]|\d+[.)])\s*", chunk):
            out.append(chunk)
            continue
        out.extend(sentences(chunk))
    return [s for s in out if len(s.split()) >= 4]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Check (c): completion-claim lexicon ──────────────────────────────────
# Restricted to service-completion verbs; generic "is used" / "is based"
# wording must never trip this (measured against TRAIN: those benign verbs
# occur 1000× more often than the completion phrases the task worries about).

_DONE_VERBS = (
    r"processed|refunded|canceled|cancelled|issued|updated|changed|"
    r"activated|deactivated|deleted|removed|switched|replaced|applied|"
    r"sent|shipped|completed|approved|resolved|recovered"
)

# passive completion: "has been processed", "has already been refunded"
_PASSIVE_BEEN = re.compile(
    rf"\b(?:has|have|had)\s+(?:already\s+|now\s+|today\s+)?been\s+"
    rf"(?:fully\s+|successfully\s+)?({_DONE_VERBS})\b",
    re.IGNORECASE,
)

# plain past tense: "was cancelled", "were shipped" (same safety rule)
_PAST_WAS = re.compile(
    rf"\b(?:was|were)\s+(?:already\s+|now\s+)?({_DONE_VERBS})\b",
    re.IGNORECASE,
)

# active completion: "I have cancelled", "we have issued", "I've processed"
_ACTIVE_CLAIM = re.compile(
    rf"\b(?:i|we)\s+(?:have|had|'ve|'d)\s+"
    rf"(?:already\s+|successfully\s+|just\s+|now\s+)?{_DONE_VERBS}\b",
    re.IGNORECASE,
)

# promise wording treated as an unverifiable completion claim
_ON_ITS_WAY_CLAIM = re.compile(r"\bon\s+its\s+way\b", re.IGNORECASE)

# Charter object nouns: order / payment / refund / account and close kin.
_MUTATION_NOUN = re.compile(
    r"\b(?:refund|reimbursement|rebate|order|account|payment|"
    r"subscription|membership|money)\b",
    re.IGNORECASE,
)
_PERSON_MARKER = re.compile(
    r"\b(?:your|yours|my|mine|you|yourself|me|we|us|our|ours|i)\b",
    re.IGNORECASE,
)


def _user_scoped(sentence: str) -> bool:
    """True when a completion phrase inside *sentence* plausibly refers to
    the user's order/payment/refund/account.

    Scoping decisions (measured against TRAIN):
    * third-person policy statements ("refunds are processed within a few
      business days after the request has been approved") have no person
      marker → not a claim for this user;
    * outgoing-communication wording ("the confirmation email was sent to
      you at the time of purchase") has no charter object noun → not a
      claim about mutating the user's order/payment/refund/account;
    * everything with a person marker AND a charter object ("your refund
      has been processed", "I have cancelled your order", "we have issued
      your refund") is treated as a completion claim.
    """
    return bool(_PERSON_MARKER.search(sentence or "")
                and _MUTATION_NOUN.search(sentence or ""))


# invented specifics: currency amounts, calendar dates, long digit ids
_AMOUNT_RE = re.compile(
    r"[$€£₹]\s?\d[\d,]*(?:\.\d{1,2})?"
    r"|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:dollars?|usd|eur|euros?|gbp|pounds?"
    r"|inr|rupees?|bucks)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b"
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?)\.?,?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b",
    re.IGNORECASE,
)
_LONG_NUM_RE = re.compile(r"\b\d{6,}\b")

# ── Placeholder neutralisation ───────────────────────────────────────────

_SLOT_LABELS = {
    "order_number": "order number",
    "invoice_number": "invoice number",
    "tracking_number": "tracking number",
    "email": "email address",
    "amount": "amount",
    "date": "date",
}


def neutral_placeholder_name(name: str) -> str:
    """`{{Order Number}}` → "your order number"; unknown names → "your <name>"."""
    slot = PLACEHOLDER_TO_SLOT.get(name.strip())
    if slot:
        return f"your {_SLOT_LABELS.get(slot, slot.replace('_', ' '))}"
    return f"your {name.strip().lower()}"


def strip_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(lambda m: neutral_placeholder_name(m.group(1)), text)


# ── Detection helpers ────────────────────────────────────────────────────

def units(text: str) -> list:
    """Bullet-aware sentence units WITHOUT the ≥4-token filter (rewriter)."""
    out: list = []
    for chunk in re.split(r"\n+", (text or "")):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.match(r"^([-*•]|\d+[.)])\s*", chunk):
            out.append(chunk)
        else:
            out.extend(sentences(chunk))
    return out


def claim_matches(text: str) -> list:
    """Completion-claim matches in *text*, sentence-scoped and user-scoped.

    A verb-form match only counts when its containing sentence indicates
    the user's order/payment/refund/account; generic policy statements and
    outgoing communication wording are exempt (see `_user_scoped`).
    """
    hits: list = []
    for unit in units(text or ""):
        if not _user_scoped(unit):
            continue
        for rx in (_PASSIVE_BEEN, _PAST_WAS, _ACTIVE_CLAIM, _ON_ITS_WAY_CLAIM):
            hits.extend(rx.finditer(unit))
    return hits


def invented_specifics(text: str, allowed_blob: str) -> list:
    """Specific values in *text* that never appear in the query or evidence.

    Placeholder tokens are ignored here: the benchmark carries
    `{{Order Number}}`-style symbols where a real user would type a value,
    and those are handled by the placeholder check (d).
    """
    plain = _PLACEHOLDER_RE.sub(" ", (text or ""))
    found: list = []
    for kind, rx in (("amount", _AMOUNT_RE), ("date", _DATE_RE),
                     ("id", _LONG_NUM_RE)):
        for m in rx.finditer(plain):
            value = m.group(0).strip()
            if value and value not in allowed_blob:
                found.append((kind, value))
    return found


# ── Aggregate helpers ────────────────────────────────────────────────────

def _evidence_pool(a2: dict) -> list:
    return [e for e in (a2.get("supporting_examples") or []) if isinstance(e, dict)]


def _evidence_intents(a2: dict, a1: dict) -> list:
    """Intent tag per supporting example; falls back to A1's prediction when
    the pipeline did not provide per-example provenance (A2 does not emit
    it) — agreement can then not be falsified against predicted intent."""
    pred = str(a1.get("intent", ""))
    return [str(ex.get("intent", pred)) for ex in _evidence_pool(a2)]


# ── The auditor ──────────────────────────────────────────────────────────

def audit(
    query: str,
    a1: dict,
    plan: Optional[RoutePlan],
    a2: dict,
    policy: Optional[dict] = None,
) -> AuditResult:
    """Audit *a2's* candidate response.  Deterministic; no LLM calls."""
    started = time.perf_counter()
    policy = policy or load_policy()

    candidate = str(a2.get("response", ""))
    predicted_intent = str(a1.get("intent", ""))
    checks: list = []
    issues: list = []
    evidence_ids = [f"ex{i}" for i in range(len(_evidence_pool(a2)))]

    def finish(verdict: str, revised: Optional[str],
               retry: Optional[int] = None) -> AuditResult:
        if retry is not None:
            issues.append(f"asked pipeline to re-audit rank-{retry} candidate")
        res = AuditResult(
            verdict=verdict,
            checks=checks,
            issues=issues,
            revised_response=revised,
            evidence_ids=evidence_ids,
            retry_candidate_rank=retry,
        )
        res.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return res

    # trivial emptiness: its own verdict, independent of the other checks
    if not candidate.strip():
        checks.append(CheckResult("empty_response", False,
                                  "A2 returned no candidate text"))
        issues.append("a2 returned an empty response; nothing to serve")
        return finish("flag", None)

    # ── (a) intent alignment ──────────────────────────────────────────
    intent_tags = _evidence_intents(a2, a1)
    needs_clar = bool(a1.get("needs_clarification", False))
    top_intents = {str(t.get("intent")) for t in (a1.get("top3") or [])}
    allowed = {predicted_intent} | (top_intents if needs_clar else set())
    misaligned = [t for t in intent_tags if t and t not in allowed]
    alignment_ok: bool = not misaligned

    sim_val: Optional[float] = None
    sim_note = ""
    try:
        from nlp.embed import embed  # reuse the existing embedding stack
        sim_min = float(get(policy, "thresholds.similarity_min", 0.70))
        qe, re_ = embed([normalize(query), normalize(candidate)])
        sim_val = float(qe @ re_)
        if sim_val < sim_min:
            alignment_ok = False
        sim_note = f", q→resp cosine {sim_val:.3f} (min {sim_min})"
    except Exception as exc:                     # embedder unavailable: skip
        sim_note = f", similarity not evaluated ({exc})"

    checks.append(CheckResult(
        "intent_alignment", alignment_ok,
        ("all evidence intents align with predicted "
         f"{predicted_intent!r}" if alignment_ok else
         f"evidence intent(s) {sorted(set(misaligned))!r} do not align with "
         f"predicted {predicted_intent!r}") + sim_note,
        value=sim_val,
    ))
    if not alignment_ok:
        issue = (f"misaligned evidence intent(s) {sorted(set(misaligned))}"
                 f" for predicted {predicted_intent!r}")
        if sim_val is not None:
            issue += f" (q→resp cosine {sim_val:.3f})"
        issues.append(issue)

    # ── (b) grounding ─────────────────────────────────────────────────
    jac_min = float(get(policy, "audit.grounding_jaccard_min", 0.35))
    ev_sents = [s for ex in _evidence_pool(a2)
                for s in auditable_sentences(str(ex.get("response", "")))]
    ev_tok_sets = [tokens(s) for s in ev_sents]
    allowed_blob = query + " \n " + " \n ".join(
        f"{ex.get('instruction', '')} {ex.get('response', '')}"
        for ex in _evidence_pool(a2))

    cands = auditable_sentences(candidate)
    ungrounded: list = []
    best_vals: list = []
    for s in cands:
        st = tokens(s)
        best = max((jaccard(st, et) for et in ev_tok_sets), default=0.0)
        best_vals.append(best)
        if best < jac_min:
            ungrounded.append(s)
    grounding_ok = (not ungrounded) if cands else True
    coverage = round(sum(best_vals) / max(len(best_vals), 1), 4)
    checks.append(CheckResult(
        "grounding", grounding_ok,
        f"{len(cands) - len(ungrounded)}/{len(cands)} sentences supported "
        f"(mean best-J={coverage})",
        value=coverage,
    ))
    if ungrounded:
        issues.append(f"{len(ungrounded)} ungrounded sentence(s): "
                      + " | ".join(u[:80] for u in ungrounded[:3]))

    # ── (c) unsupported claims / invented specifics ───────────────────
    claims = claim_matches(candidate)
    invented = invented_specifics(candidate, allowed_blob)
    claims_ok = not claims and not invented
    checks.append(CheckResult(
        "unsupported_claims", claims_ok,
        f"{len(claims)} completion claim(s), {len(invented)} invented "
        f"specific(s)",
        value=float(len(claims) + len(invented)),
    ))
    if claims:
        issues.append("completion-claim wording present: "
                      + "; ".join(f"'{m.group(0).lower()}'" for m in claims[:4]))
    if invented:
        issues.append("invented specifics not in query or evidence: "
                      + "; ".join(f"{k}='{v}'" for k, v in invented[:4]))

    # ── (d) placeholder safety ────────────────────────────────────────
    ph_names = _PLACEHOLDER_RE.findall(candidate)
    placeholders_ok = not ph_names
    checks.append(CheckResult(
        "placeholder_safety", placeholders_ok,
        "no unresolved {{...}} tokens" if placeholders_ok else
        "unresolved tokens: " + ", ".join(sorted(set(ph_names))),
        value=float(len(ph_names)),
    ))

    # ── (e) contradiction among neighbours ────────────────────────────
    # Two distinct sources: (i) conflicting procedural direction between
    # pro/con phrase axes — a text hazard no candidate swap can fix; and
    # (ii) intent disagreement across the retrieved slice — the same root
    # cause as an alignment failure, so it takes the retry path instead.
    axes_conflict, axes_detail = _axis_conflicts(predicted_intent, a2, policy)
    ratio_conflict, ratio_detail = _intent_disagreement(
        predicted_intent, intent_tags, a2, a1, policy)
    checks.append(CheckResult(
        "contradiction", not (axes_conflict or ratio_conflict),
        axes_detail if axes_conflict else ratio_detail,
        value=float(axes_conflict or ratio_conflict),
    ))
    if axes_conflict:
        issues.append("neighbour contradiction: " + axes_detail)
    elif ratio_conflict and alignment_ok:
        issues.append("neighbour contradiction: " + ratio_detail)

    # ── verdicts ──────────────────────────────────────────────────────
    multi = len(_evidence_pool(a2)) >= 2

    if axes_conflict:                              # retrying cannot fix it
        return finish("flag", None)

    if not alignment_ok:                           # wrong retrieval slice
        return finish("flag", None, retry=2 if multi else None)

    if not claims_ok or not placeholders_ok:
        revised = _rewrite(candidate, allowed_blob, policy)
        if not revised.strip():
            issues.append("revision eliminated the entire response; flagged")
            return finish("flag", None)
        if claim_matches(strip_placeholders(revised)) or \
                invented_specifics(strip_placeholders(revised), allowed_blob):
            issues.append("revision could not remove all unsafe content; flagged")
            return finish("flag", None)
        return finish("revise", revised)

    if not grounding_ok:
        if multi:                                  # ask A2 for the next best
            return finish("flag", None, retry=2)
        revised = _rewrite(candidate, allowed_blob, policy)
        revised = drop_ungrounded(revised, candidate, ev_tok_sets, jac_min)
        if not revised.strip():
            issues.append("grounded rewrite was empty; flagged")
            return finish("flag", None)
        return finish("revise", revised)

    if ratio_conflict:                             # slice disagreement alone
        return finish("flag", None)

    return finish("accept", candidate)


# ── Check (e): neighbour consistency ─────────────────────────────────────

def _intent_disagreement(predicted_intent: str, intent_tags: list,
                         a2: dict, a1: dict, policy: dict):
    """Neighbours disagree on intent (agreement below the policy minimum).

    Returns (failed, detail).
    """
    agreement_min = float(get(policy, "thresholds.neighbor_agreement_min", 0.60))
    calls = [t for t in intent_tags if t]
    if calls:
        agrees = sum(1 for t in calls if t == predicted_intent)
        ratio = agrees / len(calls)
        if ratio < agreement_min:
            return True, (f"only {agrees}/{len(calls)} evidence examples agree "
                          f"with predicted intent {predicted_intent!r} "
                          f"({ratio:.2f} < {agreement_min})")
        return False, (f"{agrees}/{len(calls)} evidence example(s) agree with "
                       f"{predicted_intent!r} ({ratio:.2f})")
    used = [str(i) for i in (a2.get("used_intents") or [])]
    if used and predicted_intent not in used:
        return True, (f"used_intents {used} do not include predicted "
                      f"intent {predicted_intent!r}")
    return False, f"evidence consistent with {predicted_intent!r}"


def _axis_conflicts(predicted_intent: str, a2: dict, policy: dict):
    """Conflicting procedural direction: both a pro- and a con- phrase of
    the same axis appear across the evidence plus the candidate itself.

    Returns (failed, detail).
    """
    ev_text = " \n ".join(str(ex.get("response", "")) for ex in _evidence_pool(a2))
    ev_text_full = ev_text + " \n " + str(a2.get("response", ""))
    hit_axes = []
    for axis in (get(policy, "audit.conflict_axes", []) or []):
        if not isinstance(axis, dict):
            continue
        pro, con = axis.get("pro"), axis.get("con")
        if pro and con:
            hit_pro = re.search(pro, ev_text_full, re.IGNORECASE)
            hit_con = re.search(con, ev_text_full, re.IGNORECASE)
            if hit_pro and hit_con:
                hit_axes.append(axis.get("axis", "unnamed"))
    if hit_axes:
        return True, ("evidence gives conflicting direction on: "
                      + ", ".join(hit_axes))
    return False, ("no conflicting procedural direction found for "
                   f"{predicted_intent!r}")


# ── Deterministic rewriter ───────────────────────────────────────────────

def _soften_passive_claim(sentence: str) -> str:
    """`has been processed` → `can be processed`: conditional wording only.
    Applied only to user-scoped sentences — generic policy statements stay."""
    # The auxiliary disappears; the sentence reads as procedural wording,
    # never completion.  "was cancelled" style plain past is softened too.
    s = re.sub(
        rf"\b(?:has|have|had)\s+(?:already\s+|now\s+|today\s+)?been\s+"
        rf"(?:fully\s+|successfully\s+)?({_DONE_VERBS})\b",
        lambda m: f"can be {m.group(1)}",
        sentence,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        rf"\b(?:was|were)\s+(?:already\s+|now\s+)?({_DONE_VERBS})\b",
        lambda m: f"can be {m.group(1)}",
        s,
        flags=re.IGNORECASE,
    )
    return s

def _drop_completed_sentences(sentence: str) -> Optional[str]:
    """None when the sentence is an active/on-its-way completion claim."""
    if _ACTIVE_CLAIM.search(sentence) or _ON_ITS_WAY_CLAIM.search(sentence):
        return None
    return sentence


def drop_ungrounded(revised: str, original: str, ev_tok_sets: list,
                    jac_min: float) -> str:
    """Keep only sentences of *original* that some evidence sentence overlaps."""
    keep: list = []
    for s in auditable_sentences(original):
        st = tokens(s)
        best = max((jaccard(st, et) for et in ev_tok_sets), default=0.0)
        if best >= jac_min:
            keep.append(s)
    return " ".join(keep) if keep else ""


def _rewrite(candidate: str, allowed_blob: str, policy: dict) -> str:
    """Deterministic repair pipeline (order matters):
       1. neutralise {{placeholders}} → "your <label>"
       2. soften passive completion claims → conditional wording
       3. drop active/on-its-way claim sentences
       4. drop sentences carrying invented specifics
    Every rewrite step re-checks the sentence; a sentence is kept only when
    it is safe.  The result must contain no completion claim when the caller
    accepts it as a `revise` — `audit` verifies that after rewriting.
    """
    out: list = []
    # bullet-aware split WITHOUT the ≥4-token filter: the rewriter must not
    # silently drop short but safe sentences ("Contact support.").
    units: list = []
    for chunk in re.split(r"\n+", candidate):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.match(r"^([-*•]|\d+[.)])\s*", chunk):
            units.append(chunk)
        else:
            units.extend(sentences(chunk))
    for s in units:
        # 1. placeholders first so checks below see concrete text
        s2 = strip_placeholders(s)
        # 2. soften passive claims
        s2 = _soften_passive_claim(s2)
        # 3. active completion claims and "on its way" cannot be softened
        s2 = _drop_completed_sentences(s2)
        if s2 is None:
            continue
        # 4. invented specifics — only drop when the value is genuinely new
        if invented_specifics(s2, allowed_blob):
            continue
        out.append(" ".join(s2.split()))
    return " ".join(out)


# ── Retry loop ───────────────────────────────────────────────────────────

def build_retry_candidate(a2: dict) -> Optional[dict]:
    """Construct the rank-2 candidate A4 wants A2's caller to serve next.

    Uses the second supporting example verbatim as both the response and its
    own provenance (a candidate comes from its source row), so the retry's
    evidence slice honestly reflects the alternated candidate.
    """
    pool = _evidence_pool(a2)
    if len(pool) < 2:
        return None
    alt = dict(a2)
    alt["response"] = str(pool[1].get("response", ""))
    alt["supporting_examples"] = [pool[1]]
    alt["used_intents"] = [str(pool[1].get("intent") or "")]
    alt["self_check"] = dict(alt.get("self_check") or {}, retry_of_rank=1)
    alt["retrieval_score"] = pool[1].get("similarity", a2.get("retrieval_score"))
    return alt


def audit_with_retry(
    query: str,
    a1: dict,
    plan: Optional[RoutePlan],
    a2: dict,
    policy: Optional[dict] = None,
) -> tuple:
    """Audit once; when the verdict asks for a rank-2 candidate, audit that
    candidate one additional time.  Returns `(final_result, retried)` where
    `retried` is True when the second pass actually ran."""
    policy = policy or load_policy()
    first = audit(query, a1, plan, a2, policy)
    if first.retry_candidate_rank != 2:
        return first, False
    alt = build_retry_candidate(a2)
    if not alt:
        return first, False
    second = audit(query, a1, plan, alt, policy)
    second.issues = [f"retry: audited rank-2 candidate (rank-1 was "
                     f"{first.verdict})"] + list(second.issues)
    return second, True
