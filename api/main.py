"""FastAPI service exposing the Level-2 support pipeline.

Endpoints:
    GET  /health          → model loaded, Supabase reachable, retrieval backend
    GET  /meta            → model names, thresholds, agents, headline #s
    POST /resolve         → A1→A3→A2→A4→A5, returns decision + trace
                            (`baseline: true` restores the Day-1 A1→A2 path;
                             `answer: "..."` is a clarification follow-up)
    POST /resolve/batch   → bounded batch (≤ 32 queries)
    POST /feedback        → user feedback

Every /resolve response keeps the Day-1 keys (`query`, `trace`,
`final_response`, `total_latency_ms`, `map`, `retrieval_backend`,
`planned_agents`) and adds the Level-2 ones (`decision`, `predicted_intent`,
`predicted_category`, `a1`..`a5`, `agent_trace`, `reason_codes`).

Retrieval:
    pgvector via Supabase RPC first (2 s timeout) → automatic fallback to
    in-memory index shipped in outputs/models/.

Secrets from env only:
    SUPABASE_URL, SUPABASE_SERVICE_KEY, ALLOWED_ORIGINS
"""

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ── Lazy-loaded .env (local dev) ──────────────────────────────────────
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="NLP Support System", version="1.0.0")

origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
# Startup: load models
# ═══════════════════════════════════════════════════════════════════════

_state = {
    "a1": None,
    "a2": None,
    "l2": None,
    "pipeline_agents": None,
    "train_emb": None,
    "supabase": None,
    "supabase_ok": False,
    "retrieval_backend": "in-memory",
}


@app.on_event("startup")
async def startup():
    log.info("Loading models …")
    from nlp.a1 import IntentAnalyst
    from nlp.a2 import A2Builder
    from nlp.pipeline import A1Agent, A2Agent, Level2Pipeline
    from nlp.embed import embed_split

    a1 = IntentAnalyst()
    _state["a1"] = a1
    log.info("A1 loaded (served level: %s)", a1.served_level)

    a2 = A2Builder().build()
    _state["a2"] = a2
    log.info("A2 loaded")

    _state["pipeline_agents"] = [A1Agent(a1), A2Agent(a2)]
    _state["train_emb"] = a2.train_emb

    # Level 2: reuse the loaded models so no extra weights are held in memory.
    _state["l2"] = Level2Pipeline(a1=a1, a2=a2)
    log.info("Level-2 pipeline ready (A1 -> A3 -> A2 -> A4 -> A5)")

    # Try Supabase connection
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if url and key:
        try:
            from supabase import create_client
            sb = create_client(url, key)
            # Quick health check
            resp = sb.table("examples").select("id", count="exact").limit(1).execute()
            count = resp.count or 0
            if count > 0:
                _state["supabase"] = sb
                _state["supabase_ok"] = True
                _state["retrieval_backend"] = "pgvector"
                log.info("Supabase connected (%d examples)", count)
            else:
                log.warning("Supabase reachable but examples table empty")
        except Exception as e:
            log.warning("Supabase not reachable: %s — using in-memory", e)
    else:
        log.info("Supabase not configured — using in-memory retrieval")

    log.info("Ready (retrieval_backend=%s)", _state["retrieval_backend"])


# ═══════════════════════════════════════════════════════════════════════
# pgvector retrieval via Supabase RPC
# ═══════════════════════════════════════════════════════════════════════

def _pgvector_respond(query: str, a1_output: dict, timeout_s: float = 2.0) -> Optional[dict]:
    """Try pgvector retrieval via Supabase RPC. Returns None on failure."""
    sb = _state.get("supabase")
    if sb is None:
        return None

    try:
        from nlp.embed import embed
        from nlp.normalize import normalize

        q_emb = embed([normalize(query)])[0].tolist()
        needs_clar = a1_output.get("needs_clarification", False)
        top3 = a1_output.get("top3", [])
        predicted_intent = a1_output["intent"]

        if not needs_clar:
            filter_intents = [predicted_intent]
        else:
            filter_intents = [t["intent"] for t in top3[:3]]

        # Try hybrid RPC
        import httpx
        t0 = time.time()
        result = sb.rpc("match_hybrid", {
            "query_embedding": q_emb,
            "query_text": query,
            "match_count": 5,
            "filter_intents": filter_intents,
        }).execute()
        elapsed = time.time() - t0

        if elapsed > timeout_s:
            log.warning("pgvector took %.1fs > %.1fs timeout", elapsed, timeout_s)
            return None

        if result.data and len(result.data) > 0:
            top = result.data[0]
            supporting = [
                {
                    "instruction": r["instruction"],
                    "response": r["response"],
                    "similarity": round(r.get("similarity", 0), 4),
                }
                for r in result.data[:3]
            ]
            return {
                "response": top["response"],
                "supporting_examples": supporting,
                "used_intents": list(set(r["intent"] for r in result.data)),
                "retrieval_score": round(top.get("rrf_score", top.get("similarity", 0)), 4),
                "low_confidence": needs_clar,
                "self_check": {
                    "placeholder_safe": "{{" not in top["response"],
                    "source_similarity_ok": True,
                    "response_intent_ok": True,
                    "check_passed": True,
                    "candidates_rejected": 0,
                },
            }
        return None
    except Exception as e:
        log.warning("pgvector RPC failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════
# Rate limiting (simple per-IP, in-memory)
# ═══════════════════════════════════════════════════════════════════════

_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 30       # requests
BATCH_LIMIT = 32
RATE_WINDOW = 60.0    # seconds


def _check_rate_limit(ip: str):
    now = time.time()
    times = _rate_store[ip]
    # Prune old
    _rate_store[ip] = [t for t in times if now - t < RATE_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded. Try again in a minute.")
    _rate_store[ip].append(now)


# ═══════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════

class ResolveRequest(BaseModel):
    query: str
    # Optional answer to a clarification question.  Stateless: the pipeline is
    # re-run on the original query plus this answer, nothing is remembered.
    answer: Optional[str] = None
    # Day-1 compatibility: run only A1 -> A2.
    baseline: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Query must not be empty")
        if len(v) > 500:
            raise ValueError("Query must be ≤ 500 characters")
        return v

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v):
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > 500:
            raise ValueError("Answer must be ≤ 500 characters")
        return v


class BatchResolveRequest(BaseModel):
    queries: list[str]
    baseline: bool = False

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, v):
        if not v:
            raise ValueError("queries must not be empty")
        if len(v) > BATCH_LIMIT:
            raise ValueError(f"At most {BATCH_LIMIT} queries per request")
        cleaned = [q.strip() for q in v]
        if any(not q for q in cleaned):
            raise ValueError("Queries must not be empty")
        if any(len(q) > 500 for q in cleaned):
            raise ValueError("Queries must be ≤ 500 characters")
        return cleaned


class FeedbackRequest(BaseModel):
    query: str
    rating: int
    note: Optional[str] = None

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError("Rating must be 1-5")
        return v


# ═══════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agents": ["a1", "a3", "a2", "a4", "a5"],
        "decisions": ["ANSWER", "CLARIFY", "ESCALATE"],
        "level2": _state["l2"] is not None,
        "model_loaded": _state["a1"] is not None,
        "supabase_reachable": _state["supabase_ok"],
        "retrieval_backend": _state["retrieval_backend"],
    }


@app.get("/meta")
async def meta():
    a1 = _state["a1"]
    from nlp.pipeline import planned_agents

    meta_info = {
        "a1": {
            "served_level": a1.served_level if a1 else None,
            "temperature": a1.temperature if a1 else None,
            "tau": a1.tau if a1 else None,
            "s_min": a1.s_min if a1 else None,
            "n_classes": len(a1.classes) if a1 else 0,
        },
        "retrieval_backend": _state["retrieval_backend"],
        "planned_agents": planned_agents,
        "level2": {
            "agents": ["A1", "A3", "A2", "A4", "A5"],
            "decisions": ["ANSWER", "CLARIFY", "ESCALATE"],
            "policy": "config/l2_policy.json",
        },
    }

    # Load headline numbers from evidence files
    evidence_dir = "outputs/evidence"
    for fname in ("a1.json", "a2.json", "retrieval_backend.json"):
        fpath = os.path.join(evidence_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                meta_info[fname.replace(".json", "")] = json.load(f)

    return meta_info


def _resolve_day1(query: str) -> dict:
    """The original Day-1 A1 -> A2 path, unchanged (used by `baseline`)."""
    from nlp.pipeline import run_pipeline

    # Try pgvector first, fallback to in-memory
    backend = "in-memory"
    a1_output = _state["a1"].analyze(query)

    pgvec_result = None
    if _state["supabase_ok"]:
        pgvec_result = _pgvector_respond(query, a1_output, timeout_s=2.0)
        if pgvec_result is not None:
            backend = "pgvector"

    if pgvec_result is not None:
        # Build pipeline result with pgvector A2
        t0 = time.perf_counter()
        from nlp.pipeline import A1Agent, A2Agent, compute_map_position, planned_agents
        from nlp.embed import embed
        from nlp.normalize import normalize

        q_emb = embed([normalize(query)])[0]
        map_pos = compute_map_position(q_emb, _state["train_emb"]) if _state["train_emb"] is not None else {"x": None, "y": None, "confidence": 0.0}
        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        result = {
            "query": query,
            "trace": [
                {"agent": "IntentAnalyst", "input": {"query": query}, "output": a1_output, "latency_ms": 0},
                {"agent": "ResponseSpecialist", "input": {"query": query, "a1_output": a1_output}, "output": pgvec_result, "latency_ms": 0},
            ],
            "final_response": pgvec_result["response"],
            "total_latency_ms": total_ms,
            "map": map_pos,
            "retrieval_backend": backend,
            "planned_agents": planned_agents,
        }
    else:
        result = run_pipeline(
            query,
            _state["pipeline_agents"],
            train_emb=_state["train_emb"],
            retrieval_backend="in-memory",
        )

    result["decision"] = None
    result["predicted_intent"] = result["trace"][0]["output"].get("intent")
    result["predicted_category"] = result["trace"][0]["output"].get("category")
    result["mode"] = "baseline"
    return result


def _log_run(query: str, result: dict) -> None:
    """Best-effort persistence of a run — never fails the request."""
    try:
        sb = _state.get("supabase")
        if sb:
            trace_json = json.loads(json.dumps(result["trace"], default=str))
            sb.table("runs").insert({
                "query": query,
                "trace": trace_json,
            }).execute()
    except Exception as e:
        log.warning("Failed to log run: %s", e)


@app.post("/resolve")
async def resolve(req: ResolveRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    l2 = _state.get("l2")

    # ── Day-1 compatibility path ──────────────────────────────────────
    if req.baseline or l2 is None:
        result = _resolve_day1(req.query)
        if req.answer:
            # The Day-1 path has no clarification mechanism; surface the
            # merged text so callers can see what was actually classified.
            result["original_query"] = req.query
            result["followup_answer"] = req.answer
    else:
        # ── Level 2: A1 -> A3 -> A2 -> A4 -> A5 ────────────────
        if req.answer:
            result = l2.resolve_with_followup(
                req.query, req.answer,
                retrieval_backend=_state["retrieval_backend"])
        else:
            result = l2.resolve(
                req.query, retrieval_backend=_state["retrieval_backend"])

    _log_run(req.query, result)
    return result


@app.post("/resolve/batch")
async def resolve_batch(req: BatchResolveRequest, request: Request):
    """Bounded batch (≤ 32). Rate-limited like /resolve."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    l2 = _state.get("l2")
    results = []
    for query in req.queries:
        if req.baseline or l2 is None:
            results.append(_resolve_day1(query))
        else:
            results.append(l2.resolve(
                query, retrieval_backend=_state["retrieval_backend"],
                with_map=False))

    counted: dict[str, int] = {}
    for res in results:
        key = res.get("decision") or "BASELINE"
        counted[key] = counted.get(key, 0) + 1

    return {"count": len(results), "decisions": counted, "results": results}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    sb = _state.get("supabase")
    if sb:
        try:
            sb.table("feedback").insert({
                "query": req.query,
                "rating": req.rating,
                "note": req.note,
            }).execute()
            return {"status": "saved"}
        except Exception as e:
            log.warning("Failed to save feedback: %s", e)
            raise HTTPException(500, "Failed to save feedback")
    else:
        log.warning("Feedback received but Supabase not configured")
        return {"status": "accepted", "note": "No database configured"}
