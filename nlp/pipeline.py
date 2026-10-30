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
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(
    query: str,
    agents: list[Agent],
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
