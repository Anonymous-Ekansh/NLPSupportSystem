"""Orchestrator pipeline: Agent interface + trace recording.

Runs [A1, A2] sequentially, records trace steps with latency,
and exposes planned Level-2 agents as placeholders.

Returns:
    {
        "query", "trace", "final_response", "total_latency_ms",
        "map": {"x", "y", "confidence"},
        "retrieval_backend"
    }
"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

# Level-2 agent names, in execution order.  A2/A4 are skipped (and their trace
# step marked "skipped") when A3 already decided clarify or escalate, so every
# result carries the same five steps.
L2_AGENT_ORDER = ("A1", "A3", "A2", "A4", "A5")


# ═══════════════════════════════════════════════════════════════════════
# Agent interface
# ═══════════════════════════════════════════════════════════════════════

class Agent(ABC):
    """Base class for all pipeline agents."""

    name: str = "unnamed"

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Execute the agent, mutating and returning *state*."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Concrete agents wrapping A1, A2
# ═══════════════════════════════════════════════════════════════════════

class A1Agent(Agent):
    """Wraps IntentAnalyst.analyze into the Agent interface."""

    name = "IntentAnalyst"

    def __init__(self, analyst):
        self._a = analyst

    def run(self, state: dict) -> dict:
        a1_out = self._a.analyze(state["query"])
        state["a1_output"] = a1_out
        return state


class A2Agent(Agent):
    """Wraps ResponseSpecialist.respond into the Agent interface."""

    name = "ResponseSpecialist"

    def __init__(self, specialist, mode: str = "hybrid"):
        self._s = specialist
        self._mode = mode

    def run(self, state: dict) -> dict:
        a2_out = self._s.respond(
            state["query"],
            state["a1_output"],
            mode=self._mode,
        )
        state["a2_output"] = a2_out
        return state


# ═══════════════════════════════════════════════════════════════════════
# Planned Level-2 agents (not yet executed)
# ═══════════════════════════════════════════════════════════════════════

planned_agents = [
    {"name": "Verifier",   "status": "planned",
     "description": "Cross-checks A2 response against policy knowledge base"},
    {"name": "Clarifier",  "status": "planned",
     "description": "Generates follow-up questions when confidence is low"},
    {"name": "Escalator",  "status": "planned",
     "description": "Routes to human agents based on severity and topic"},
]


# ═══════════════════════════════════════════════════════════════════════
# Map projection helper
# ═══════════════════════════════════════════════════════════════════════

_MAP_POINTS = None

def _load_map_points() -> Optional[dict]:
    """Load precomputed map coordinates (lazy, singleton)."""
    global _MAP_POINTS
    if _MAP_POINTS is not None:
        return _MAP_POINTS
    path = "outputs/map_points.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        _MAP_POINTS = json.load(f)
    return _MAP_POINTS


def compute_map_position(q_emb: np.ndarray, train_emb: np.ndarray) -> dict:
    """Similarity-weighted mean of 10 nearest neighbours' coordinates.

    Args:
        q_emb:     (384,) L2-normed query embedding.
        train_emb: (N, 384) L2-normed TRAIN embeddings.

    Returns:
        {"x": float|null, "y": float|null, "confidence": float}
    """
    pts = _load_map_points()
    if pts is None:
        return {"x": None, "y": None, "confidence": 0.0}

    coords = pts.get("coordinates", [])
    if not coords:
        return {"x": None, "y": None, "confidence": 0.0}

    # Build arrays of (x, y) and embeddings for points that have coords
    xs = np.array([c["x"] for c in coords], dtype=np.float32)
    ys = np.array([c["y"] for c in coords], dtype=np.float32)
    indices = np.array([c["index"] for c in coords], dtype=np.int64)

    # Similarities to those points
    sims = train_emb[indices] @ q_emb  # (M,)
    top10 = np.argsort(sims)[-10:]
    top_sims = sims[top10]
    top_sims = np.maximum(top_sims, 0)  # clip negatives

    w = top_sims / (top_sims.sum() + 1e-9)
    x = float(np.dot(w, xs[top10]))
    y = float(np.dot(w, ys[top10]))
    confidence = float(top_sims.mean())

    return {"x": round(x, 4), "y": round(y, 4), "confidence": round(confidence, 4)}


# ═══════════════════════════════════════════════════════════════════════
# Level 2 orchestrator: A1 -> A3 -> A2 -> A4 -> (A2 retry -> A4) -> A5
# ═══════════════════════════════════════════════════════════════════════

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_path(*parts: str) -> str:
    """Resolve a repo-relative path so the pipeline works from any cwd."""
    return os.path.join(_REPO_ROOT, *parts)


class Level2Pipeline:
    """One entry point returning exactly one decision plus a 5-agent trace.

    Day-1 behaviour is preserved twice over:
      * `resolve(..., baseline=True)` runs the original A1 -> A2 pipeline
        (`run_pipeline`) and returns its dict shape unchanged;
      * the Level-2 path returns the same Day-1 keys (`final_response`,
        `trace`, `map`, `retrieval_backend`, `total_latency_ms`) alongside the
        new ones, so existing clients keep working.

    Heavy models are built lazily and shared, so a warm call is one A1 pass,
    one A2 retrieval and one audit — no model loading.
    """

    def __init__(self, a1=None, a2=None, planner=None, policy_path=None):
        self._a1 = a1
        self._a2 = a2
        self._planner = planner
        self._policy_path = policy_path
        self._intent_of_instruction: Optional[dict] = None

    # ── lazy model access ────────────────────────────────────────────

    @property
    def a1(self):
        if self._a1 is None:
            from nlp.a1 import IntentAnalyst
            self._a1 = IntentAnalyst()
        return self._a1

    @property
    def a2(self):
        if self._a2 is None:
            from nlp.a2 import A2Builder
            self._a2 = A2Builder().build()
        return self._a2

    @property
    def planner(self):
        if self._planner is None:
            from nlp.l2.a3 import RoutePlanner
            self._planner = RoutePlanner(policy_path=self._policy_path)
        return self._planner

    @property
    def train_emb(self):
        return getattr(self.a2, "train_emb", None)

    def _annotate_evidence(self, a2_out: dict) -> dict:
        """Attach each supporting example's dataset intent.

        A2 does not emit per-example provenance, but A4's alignment and
        contradiction checks need it.  The lookup is a one-off dict built from
        A2's own TRAIN frame, so it costs nothing per request.
        """
        if self._intent_of_instruction is None:
            try:
                df = self.a2.train_df
                self._intent_of_instruction = {
                    str(r.instruction): str(r.intent) for r in df.itertuples()
                }
            except Exception:
                self._intent_of_instruction = {}
        for ex in a2_out.get("supporting_examples") or []:
            if not ex.get("intent"):
                tag = self._intent_of_instruction.get(
                    str(ex.get("instruction", "")))
                if tag:
                    ex["intent"] = tag
        return a2_out

    @staticmethod
    def _a1_for_a2(a1_output: dict, plan) -> dict:
        """Hand A2 the retrieval scope A3 chose.

        A2 keys its intent filter off `needs_clarification`; A3 already
        decided whether to stay on top-1 or widen.  Keeping the two in sync
        means the plan actually governs retrieval instead of being advisory.
        """
        scoped = dict(a1_output)
        intents = list(getattr(plan.retrieval, "intents", []) or [])
        scoped["needs_clarification"] = len(intents) > 1
        return scoped

    # ── main entry point ────────────────────────────────────────────

    def resolve(
        self,
        query: str,
        baseline: bool = False,
        mode: Optional[str] = None,
        retrieval_backend: str = "in-memory",
        with_map: bool = True,
        a1_override: Optional[dict] = None,
    ) -> dict:
        """Run the pipeline once.  `baseline=True` restores the Day-1 path.

        `a1_override` is used for clarification follow-ups: the intent is
        classified from the original request while the rest of the run sees
        the query plus the answer (so the slot detector can find it).
        """
        query = (query or "").strip()

        # ── Day 1: A1 -> A2, byte-for-byte the original orchestrator ──
        if baseline:
            result = run_pipeline(
                query,
                [A1Agent(self.a1), A2Agent(self.a2)],
                train_emb=self.train_emb,
                retrieval_backend=retrieval_backend,
            )
            result["mode"] = "baseline"
            result["decision"] = None
            result["predicted_intent"] = (result.get("trace") or [{}])[0].get(
                "output", {}).get("intent")
            return result

        from nlp.l2.a4 import audit, build_retry_candidate
        from nlp.a5 import coordinate

        lat: dict = {}
        t_all = time.perf_counter()

        # ── A1 intent ──────────────────────────────────────────────
        t0 = time.perf_counter()
        a1_output = a1_override if a1_override is not None \
            else self.a1.analyze(query)
        lat["a1"] = round((time.perf_counter() - t0) * 1000, 1)

        # ── A3 route ───────────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self.planner.plan(query, a1_output, analyzer=self.a1)
        lat["a3"] = round((time.perf_counter() - t0) * 1000, 1)

        a2_output = None
        audit_result = None
        retry_info: dict = {"retried": False, "latency_ms": lat}

        # A2/A4 only run on the answer path; clarify/escalate need no candidate.
        if plan.route == "answer":
            t0 = time.perf_counter()
            try:
                a2_output = self.a2.respond(
                    query,
                    self._a1_for_a2(a1_output, plan),
                    mode=mode or plan.retrieval.mode,
                    k=int(plan.retrieval.k),
                )
                a2_output = self._annotate_evidence(a2_output)
            except Exception as exc:                 # never lose the decision
                retry_info["a2_error"] = f"{type(exc).__name__}: {exc}"
                a2_output = None
            lat["a2"] = round((time.perf_counter() - t0) * 1000, 1)

            if a2_output is not None:
                # ── A4 audit, with at most one rank-2 retry ──────────
                t0 = time.perf_counter()
                first = audit(query, a1_output, plan, a2_output)
                lat["a4"] = round((time.perf_counter() - t0) * 1000, 1)
                audit_result = first

                if first.retry_candidate_rank == 2:
                    alt = build_retry_candidate(a2_output)
                    if alt is not None:
                        alt = self._annotate_evidence(alt)
                        t0 = time.perf_counter()
                        second = audit(query, a1_output, plan, alt)
                        lat["a4_retry"] = round(
                            (time.perf_counter() - t0) * 1000, 1)
                        second.issues = [
                            "retry: audited rank-2 candidate "
                            f"(rank-1 was {first.verdict})"] + list(second.issues)
                        retry_info.update({
                            "retried": True,
                            "first_audit": first,
                            "a2_first": a2_output,
                            "a2_retry": alt,
                        })
                        audit_result = second
                        a2_output = alt

        # ── A5 decision + trace ────────────────────────────────────
        retry_info["latency_ms"] = lat
        t0 = time.perf_counter()
        decision, steps = coordinate(
            query, a1_output, plan, a2_output, audit_result, retry_info)
        lat["a5"] = round((time.perf_counter() - t0) * 1000, 1)
        retry_info["latency_ms"] = lat
        # The A5 step was stamped before its own latency existed; restamp it.
        if steps and steps[-1].agent == "A5":
            steps[-1].latency_ms = lat["a5"]

        total_ms = round((time.perf_counter() - t_all) * 1000, 1)

        # ── map projection (unchanged from Day 1) ──────────────────
        map_pos = {"x": None, "y": None, "confidence": 0.0}
        if with_map and self.train_emb is not None:
            try:
                from nlp.embed import embed
                from nlp.normalize import normalize
                q_emb = embed([normalize(query)])[0]
                map_pos = compute_map_position(q_emb, self.train_emb)
            except Exception:
                pass

        trace = [_step_to_dict(s) for s in steps]

        return {
            "query": query,
            "decision": decision.decision,
            "response": decision.response,
            "final_response": decision.response,
            "predicted_intent": decision.intent,
            "predicted_category": decision.category,
            "confidence": a1_output.get("confidence"),
            "reason_codes": list(decision.reason_codes),
            "evidence_ids": list(decision.evidence_ids),
            "trace": trace,
            "agent_trace": [s.to_dict() for s in steps],
            "a1": a1_output,
            "a3": plan.to_dict(),
            "a2": a2_output,
            "a4": audit_result.to_dict() if audit_result else None,
            "a5": decision.to_dict(),
            "map": map_pos,
            "total_latency_ms": total_ms,
            "latency_breakdown_ms": lat,
            "retrieval_backend": retrieval_backend,
            "mode": "l2",
            "planned_agents": planned_agents,
        }


    def resolve_with_followup(
        self,
        query: str,
        answer: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Stateless clarification follow-up.

        A clarification is answered by re-running the whole pipeline on the
        original query plus the answer ("I need a refund but cannot find my
        order number" + "123456"), so the slot detector sees both.  No server
        state is kept between calls.
        """
        answer = (answer or "").strip()
        if not answer:
            return self.resolve(query, **kwargs)
        merged = f"{query.strip().rstrip('.')} {answer}"
        # Intent comes from the original request: appending a bare order number
        # must supply the missing slot, not change what the user asked for.
        a1_original = self.a1.analyze(query)
        result = self.resolve(merged, a1_override=a1_original, **kwargs)
        result["original_query"] = query
        result["followup_answer"] = answer
        result["merged_query"] = merged
        return result


def _step_to_dict(step) -> dict:
    """Trace step as a dict that is a superset of the Day-1 step shape."""
    return {
        "agent": step.agent,
        "role": step.role,
        "input": step.payload.get("input", {}),
        "output": step.payload.get("output", {}),
        "decision": step.decision,
        "evidence": list(step.evidence),
        "input_summary": step.input_summary,
        "output_summary": step.output_summary,
        "latency_ms": step.latency_ms,
    }


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(
    query: str,
    agents: list,
    train_emb: Optional[np.ndarray] = None,
    retrieval_backend: str = "in-memory",
) -> dict:
    """Execute the agent pipeline and return the full result dict.

    Args:
        query:             The user query string.
        agents:            Ordered list of Agent instances to execute.
        train_emb:         (N, 384) TRAIN embeddings for map projection.
        retrieval_backend: "pgvector" or "in-memory".

    Returns:
        {
            "query", "trace", "final_response", "total_latency_ms",
            "map": {"x", "y", "confidence"},
            "retrieval_backend"
        }
    """
    state = {"query": query}
    trace = []
    t_total = time.perf_counter()

    for agent in agents:
        step_input = dict(state)  # snapshot
        t0 = time.perf_counter()
        state = agent.run(state)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        trace.append({
            "agent": agent.name,
            "input": step_input,
            "output": state.get(f"{agent.name.lower()}_output",
                                state.get("a1_output") if agent.name == "IntentAnalyst"
                                else state.get("a2_output")),
            "latency_ms": latency_ms,
        })

    total_latency_ms = round((time.perf_counter() - t_total) * 1000, 1)

    # Map position
    map_pos = {"x": None, "y": None, "confidence": 0.0}
    if train_emb is not None and "a1_output" in state:
        from nlp.embed import embed
        from nlp.normalize import normalize
        q_emb = embed([normalize(query)])[0]
        map_pos = compute_map_position(q_emb, train_emb)

    # Final response
    a2_out = state.get("a2_output", {})
    final_response = a2_out.get("response", "")

    return {
        "query": query,
        "trace": trace,
        "final_response": final_response,
        "total_latency_ms": total_latency_ms,
        "map": map_pos,
        "retrieval_backend": retrieval_backend,
        "planned_agents": planned_agents,
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI (single query).  `--baseline` restores the Day-1 A1 -> A2 path.
# ═══════════════════════════════════════════════════════════════════════

def _format_trace(result: dict) -> str:
    """Human-readable five-agent trace."""
    lines = []
    head = result.get("decision") or "BASELINE"
    lines.append(f"{'=' * 68}")
    lines.append(f"{head}  |  {result.get('predicted_intent') or '-'}"
                 f" / {result.get('predicted_category') or '-'}")
    lines.append(f"{'=' * 68}")
    lines.append(result.get("final_response", ""))
    lines.append("")
    lines.append(f"{'agent':<4} {'role':<24} {'decision':<18} "
                 f"{'ms':>7}  evidence")
    lines.append("-" * 68)
    for step in result.get("trace") or []:
        agent = str(step.get("agent", ""))[:4]
        role = str(step.get("role", ""))[:24]
        dec = str(step.get("decision", ""))[:18]
        ms = f"{float(step.get('latency_ms') or 0):.1f}"
        ev = ", ".join(str(e) for e in (step.get("evidence") or [])[:3])
        lines.append(f"{agent:<4} {role:<24} {dec:<18} {ms:>7}  {ev}")
    lines.append("-" * 68)
    lines.append(f"total {float(result.get('total_latency_ms') or 0):.1f} ms"
                 f"   reasons: {', '.join(result.get('reason_codes') or [])}")
    return "\n".join(lines)


def _main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the support pipeline once.")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--baseline", action="store_true",
                    help="run the Day-1 A1 -> A2 path only")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args(argv)
    query = " ".join(args.query)

    pipe = Level2Pipeline()
    result = pipe.resolve(query, baseline=args.baseline)
    print(json.dumps(result, indent=2, default=str) if args.json
          else _format_trace(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
