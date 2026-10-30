-- ╔══════════════════════════════════════════════════════════════════════╗
-- ║  NLP Support System — Supabase Schema                               ║
-- ║  Paste this ENTIRE file into the Supabase SQL Editor and run once.  ║
-- ╚══════════════════════════════════════════════════════════════════════╝

-- ① Enable the pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ═══════════════════════════════════════════════════════════════════════
-- TABLES
-- ═══════════════════════════════════════════════════════════════════════

-- ── examples ─────────────────────────────────────────────────────────
-- Each row = one TRAIN instruction/response pair with its embedding.
CREATE TABLE IF NOT EXISTS examples (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    intent      TEXT        NOT NULL,
    category    TEXT        NOT NULL,
    flags       TEXT,
    instruction TEXT        NOT NULL,
    response    TEXT        NOT NULL,
    embedding   vector(384) NOT NULL,
    map_x       REAL,                       -- UMAP / t-SNE x coordinate
    map_y       REAL,                       -- UMAP / t-SNE y coordinate
    fts         TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', instruction)
                ) STORED
);

-- HNSW index on cosine distance for fast ANN search
CREATE INDEX IF NOT EXISTS idx_examples_embedding_hnsw
    ON examples
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index on full-text search column
CREATE INDEX IF NOT EXISTS idx_examples_fts_gin
    ON examples USING gin (fts);

-- B-tree on intent for filtered queries
CREATE INDEX IF NOT EXISTS idx_examples_intent
    ON examples (intent);


-- ── runs ─────────────────────────────────────────────────────────────
-- Every /resolve call is logged here.
CREATE TABLE IF NOT EXISTS runs (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now() NOT NULL,
    query       TEXT        NOT NULL,
    trace       JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at DESC);


-- ── feedback ─────────────────────────────────────────────────────────
-- User feedback submitted via /feedback.
CREATE TABLE IF NOT EXISTS feedback (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now() NOT NULL,
    query       TEXT        NOT NULL,
    rating      SMALLINT    NOT NULL CHECK (rating BETWEEN 1 AND 5),
    note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback (created_at DESC);


-- ═══════════════════════════════════════════════════════════════════════
-- ROW LEVEL SECURITY (on, no public policies → service_role only)
-- ═══════════════════════════════════════════════════════════════════════

ALTER TABLE examples ENABLE ROW LEVEL SECURITY;
ALTER TABLE runs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- No policies → only service_role key (which bypasses RLS) can read/write.


-- ═══════════════════════════════════════════════════════════════════════
-- RPC FUNCTIONS
-- ═══════════════════════════════════════════════════════════════════════

-- ── match_vector ─────────────────────────────────────────────────────
-- Pure vector similarity search with optional intent filter.
CREATE OR REPLACE FUNCTION match_vector(
    query_embedding  vector(384),
    match_count      INT DEFAULT 5,
    filter_intents   TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id          BIGINT,
    intent      TEXT,
    category    TEXT,
    flags       TEXT,
    instruction TEXT,
    response    TEXT,
    similarity  FLOAT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.intent,
        e.category,
        e.flags,
        e.instruction,
        e.response,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM examples e
    WHERE
        CASE
            WHEN filter_intents IS NOT NULL
            THEN e.intent = ANY(filter_intents)
            ELSE TRUE
        END
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- ── match_hybrid ─────────────────────────────────────────────────────
-- Reciprocal Rank Fusion of vector similarity + full-text search.
-- The query_text is sanitised: strip {{...}}, punctuation, collapse spaces.
CREATE OR REPLACE FUNCTION match_hybrid(
    query_embedding  vector(384),
    query_text       TEXT,
    match_count      INT DEFAULT 5,
    filter_intents   TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id          BIGINT,
    intent      TEXT,
    category    TEXT,
    flags       TEXT,
    instruction TEXT,
    response    TEXT,
    similarity  FLOAT,
    rrf_score   FLOAT
)
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    k_rrf    CONSTANT INT := 60;
    pool     CONSTANT INT := 200;   -- fetch top-N from each leg
    -- Sanitise: strip {{...}}, non-alphanumeric (keep spaces), collapse
    sanitized TEXT := regexp_replace(
                        regexp_replace(query_text, '\{\{[^}]*\}\}', '', 'g'),
                        '[^a-zA-Z0-9 ]', ' ', 'g');
BEGIN
    sanitized := regexp_replace(trim(sanitized), '\s+', ' ', 'g');

    RETURN QUERY
    WITH
    -- Leg 1: vector similarity (cosine)
    vec AS (
        SELECT
            e.id                                              AS eid,
            ROW_NUMBER() OVER (ORDER BY e.embedding <=> query_embedding) AS rn
        FROM examples e
        WHERE CASE WHEN filter_intents IS NOT NULL
                   THEN e.intent = ANY(filter_intents) ELSE TRUE END
        ORDER BY e.embedding <=> query_embedding
        LIMIT pool
    ),
    -- Leg 2: full-text search (ts_rank_cd)
    fts AS (
        SELECT
            e.id                                              AS eid,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(e.fts,
                websearch_to_tsquery('english', sanitized)) DESC) AS rn
        FROM examples e
        WHERE
            e.fts @@ websearch_to_tsquery('english', sanitized)
            AND CASE WHEN filter_intents IS NOT NULL
                     THEN e.intent = ANY(filter_intents) ELSE TRUE END
        ORDER BY ts_rank_cd(e.fts,
            websearch_to_tsquery('english', sanitized)) DESC
        LIMIT pool
    ),
    -- Combine with RRF
    combined AS (
        SELECT
            COALESCE(v.eid, f.eid)                            AS eid,
            COALESCE(1.0 / (k_rrf + v.rn), 0)
                + COALESCE(1.0 / (k_rrf + f.rn), 0)          AS rrf
        FROM vec v
        FULL OUTER JOIN fts f ON v.eid = f.eid
    )
    SELECT
        e.id,
        e.intent,
        e.category,
        e.flags,
        e.instruction,
        e.response,
        (1 - (e.embedding <=> query_embedding))::FLOAT        AS similarity,
        c.rrf::FLOAT                                           AS rrf_score
    FROM combined c
    JOIN examples e ON e.id = c.eid
    ORDER BY c.rrf DESC
    LIMIT match_count;
END;
$$;
