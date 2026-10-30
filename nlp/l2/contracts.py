"""Typed contracts shared by the Level-2 agents (A3 route, A4 audit, A5 coordinate).

Plain dataclasses (no new dependency; the repo already uses pydantic only in
`api/`, and these objects must stay cheap to build per request).

Every contract exposes `to_dict()` so it can be dropped straight into a
`/resolve` JSON response / trace step.

Design notes
------------
* `RoutePlan.route` uses the internal lowercase route names
  (`answer` / `clarify` / `escalate`); `FinalDecision.decision` uses the
  user-facing uppercase outcome (`ANSWER` / `CLARIFY` / `ESCALATE`).
  Keeping the two vocabularies separate means A3 can widen retrieval and still
  end up `ANSWER`, without a silent vocabulary clash with A5.
* `reasons` are human-readable sentences shown in the trace; `reason_codes`
  are stable snake_case tokens for scripted evaluation. Both are kept: the UI
  needs prose, the tuning/eval scripts need codes.
* Fields are additive-only. Day 1 trace dicts (`agent`, `input`, `output`,
  `latency_ms`) stay valid; `TraceStep` carries the richer Level-2 shape.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ── Vocabularies ────────────────────────────────────────────────────────
ROUTES = ("answer", "clarify", "escalate")
VERDICTS = ("accept", "revise", "flag")
DECISIONS = ("ANSWER", "CLARIFY", "ESCALATE")


# ── A3: route plan ──────────────────────────────────────────────────────

@dataclass
class MissingSlot:
    """A required piece of information the user has not (usably) supplied."""

    slot: str                      # canonical slot id, e.g. "order_number"
    label: str                     # human label, e.g. "order number"
    question: str                  # ready-to-send follow-up question
    evidence: str = ""             # why we think it is missing (matched phrase)
    required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClarifyOption:
    """One candidate intent offered back to the user on the clarify path."""

    intent: str
    prob: float
    question: str
    category: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalSettings:
    """What A2 should do, decided by A3 (A2 currently takes mode/k only)."""

    mode: str = "hybrid"
    k: int = 5
    intents: list[str] = field(default_factory=list)
    widen: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoutePlan:
    """A3 output: what happens next, and why."""

    route: str                                     # one of ROUTES
    reasons: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    missing_slots: list[MissingSlot] = field(default_factory=list)
    clarify_options: list[ClarifyOption] = field(default_factory=list)
    intent: str = ""                                # A1 intent the plan is built on
    category: str = ""
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {self.route!r}")

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "reasons": list(self.reasons),
            "reason_codes": list(self.reason_codes),
            "retrieval": self.retrieval.to_dict(),
            "missing_slots": [m.to_dict() for m in self.missing_slots],
            "clarify_options": [c.to_dict() for c in self.clarify_options],
            "intent": self.intent,
            "category": self.category,
            "confidence": self.confidence,
        }


# ── A4: audit ───────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Outcome of one named audit check."""

    name: str
    passed: bool
    detail: str = ""
    value: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditResult:
    """A4 output: accept the draft, revise it, or flag it for a human."""

    verdict: str                                   # one of VERDICTS
    checks: list[CheckResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    revised_response: Optional[str] = None
    evidence_ids: list[str] = field(default_factory=list)
    retry_candidate_rank: Optional[int] = None     # 1-based rank to ask A2 for

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "checks": [c.to_dict() for c in self.checks],
            "issues": list(self.issues),
            "revised_response": self.revised_response,
            "evidence_ids": list(self.evidence_ids),
            "retry_candidate_rank": self.retry_candidate_rank,
        }


# ── A5: final decision ──────────────────────────────────────────────────

@dataclass
class FinalDecision:
    """A5 output: the one answer returned to the user."""

    decision: str                                  # one of DECISIONS
    response: str
    intent: str = ""
    category: str = ""
    reason_codes: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(
                f"decision must be one of {DECISIONS}, got {self.decision!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# ── Trace ───────────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    """One agent step in the Level-2 trace.

    Additive to the Day 1 step shape (`agent`, `input`, `output`,
    `latency_ms`): `role`, `decision`, `evidence` and the summaries are new.
    """

    agent: str
    role: str = ""
    input_summary: str = ""
    output_summary: str = ""
    decision: str = ""
    evidence: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_legacy(self) -> dict:
        """Day-1-compatible step: same four keys the current UI reads."""
        return {
            "agent": self.agent,
            "input": {"summary": self.input_summary},
            "output": self.payload,
            "latency_ms": self.latency_ms,
        }


__all__ = [
    "ROUTES", "VERDICTS", "DECISIONS",
    "MissingSlot", "ClarifyOption", "RetrievalSettings", "RoutePlan",
    "CheckResult", "AuditResult", "FinalDecision", "TraceStep",
]
