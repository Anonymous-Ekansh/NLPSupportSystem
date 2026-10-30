"""FastAPI service exposing the two-agent NLP pipeline.

Endpoints:
    GET  /health         → {"status": "ok", "agents": ["a1", "a2"]}
    POST /query          → full A1→A2 pipeline result
    POST /a1/analyze     → A1 only
    POST /a2/respond     → A2 only (requires A1 output in body)
    GET  /metrics        → latest test metrics
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Pydantic models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="Customer support query")

class A2Request(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    a1_output: dict = Field(..., description="Output from A1.analyze()")

class BatchQueryRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=50,
                               description="List of queries (max 50)")


# ── Global agent references ──────────────────────────────────────────

_a1 = None
_a2 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup."""
    global _a1, _a2
    print("[api] Loading A1 (Intent Analyst) …")
    from nlp.a1 import IntentAnalyst
    _a1 = IntentAnalyst()
    print("[api] Loading A2 (Response Specialist) …")
    from nlp.a2 import A2Builder
    _a2 = A2Builder(embed_model=_a1._embed_model).build()
    print("[api] Ready.")
    yield
    print("[api] Shutting down.")


app = FastAPI(
    title="NLP Support System API",
    description="Two-agent customer support pipeline: A1 (Intent Analyst) → A2 (Response Specialist)",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for the Next.js frontend (or any origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check – used by Render, UptimeRobot, and Vercel."""
    return {
        "status": "ok",
        "agents": ["a1", "a2"],
        "a1_loaded": _a1 is not None,
        "a2_loaded": _a2 is not None,
    }


@app.post("/query")
async def query(req: QueryRequest):
    """Full pipeline: query → A1 → A2 → structured response."""
    if _a1 is None or _a2 is None:
        raise HTTPException(503, "Models not loaded yet")
    t0 = time.time()
    a1_out = _a1.analyze(req.query)
    a2_out = _a2.respond(req.query, a1_out)
    elapsed = round(time.time() - t0, 4)
    return {
        "query": req.query,
        "a1_output": a1_out,
        "a2_output": a2_out,
        "latency_s": elapsed,
    }


@app.post("/a1/analyze")
async def a1_analyze(req: QueryRequest):
    """A1 only – intent classification."""
    if _a1 is None:
        raise HTTPException(503, "A1 not loaded yet")
    t0 = time.time()
    result = _a1.analyze(req.query)
    result["latency_s"] = round(time.time() - t0, 4)
    return result


@app.post("/a2/respond")
async def a2_respond(req: A2Request):
    """A2 only – given a query + A1 output, return response."""
    if _a2 is None:
        raise HTTPException(503, "A2 not loaded yet")
    t0 = time.time()
    result = _a2.respond(req.query, req.a1_output)
    result["latency_s"] = round(time.time() - t0, 4)
    return result


@app.post("/batch")
async def batch_query(req: BatchQueryRequest):
    """Batch query – process up to 50 queries in one call."""
    if _a1 is None or _a2 is None:
        raise HTTPException(503, "Models not loaded yet")
    t0 = time.time()
    results = []
    for q in req.queries:
        a1_out = _a1.analyze(q)
        a2_out = _a2.respond(q, a1_out)
        results.append({
            "query": q,
            "a1_output": a1_out,
            "a2_output": a2_out,
        })
    elapsed = round(time.time() - t0, 4)
    return {"results": results, "count": len(results), "latency_s": elapsed}


@app.get("/metrics")
async def metrics():
    """Return the latest test metrics."""
    path = "outputs/metrics/test_metrics.json"
    if not os.path.exists(path):
        raise HTTPException(404, "No metrics found – run scripts/run_all.py first")
    with open(path) as f:
        return json.load(f)


@app.get("/examples")
async def examples():
    """Return precomputed example traces."""
    path = "outputs/examples/traces.json"
    if not os.path.exists(path):
        raise HTTPException(404, "No examples found – run scripts/run_all.py first")
    with open(path) as f:
        return json.load(f)
