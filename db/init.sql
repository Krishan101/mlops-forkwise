-- ForkWise MLOps Platform — PostgreSQL Schema
-- Loaded automatically on first container start via /docker-entrypoint-initdb.d/

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- recipe_metadata
-- Populated by the ingest service when recipes arrive from Mealie.
-- Primary reference linking recipe_id to its source and ingestion provenance.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recipe_metadata (
    recipe_id       VARCHAR(255) PRIMARY KEY,
    name            TEXT         NOT NULL,
    slug            VARCHAR(255),
    source          VARCHAR(50)  NOT NULL DEFAULT 'mealie',
    ingredient_count INTEGER     NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- recipe_ingredients
-- Normalized ingredient list per recipe, used for embedding and substitution.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recipe_ingredients (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    recipe_id       VARCHAR(255) NOT NULL REFERENCES recipe_metadata(recipe_id) ON DELETE CASCADE,
    ingredient      TEXT         NOT NULL,
    position        INTEGER      NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_recipe_ingredients_recipe_id ON recipe_ingredients(recipe_id);

-- ---------------------------------------------------------------------------
-- substitution_queries
-- One row per user substitution request.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS substitution_queries (
    query_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    recipe_id       VARCHAR(255),
    original_ingredient TEXT     NOT NULL,
    query_context   TEXT,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- substitution_results
-- Ranked suggestions returned for each query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS substitution_results (
    result_id       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id        UUID         NOT NULL REFERENCES substitution_queries(query_id) ON DELETE CASCADE,
    suggested_ingredient TEXT    NOT NULL,
    rank            INTEGER      NOT NULL CHECK (rank >= 1),
    score           FLOAT,
    accepted        INTEGER      NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sub_results_query_id ON substitution_results(query_id);

-- ---------------------------------------------------------------------------
-- feedback_events
-- User accepts/rejects a substitution suggestion.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feedback_events (
    event_id        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id        UUID         NOT NULL REFERENCES substitution_queries(query_id) ON DELETE CASCADE,
    suggested_ingredient TEXT    NOT NULL,
    event_type      VARCHAR(20)  NOT NULL CHECK (event_type IN ('accept', 'reject')),
    rank            INTEGER      NOT NULL CHECK (rank >= 1),
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_events_query_id ON feedback_events(query_id);

-- ---------------------------------------------------------------------------
-- feature_jobs
-- Async job queue for computing ingredient embeddings.
-- Ingest service creates a job; feature-worker picks it up and writes to Qdrant.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_jobs (
    job_id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    recipe_id       VARCHAR(255) NOT NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'pending'
                                 CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_feature_jobs_status ON feature_jobs(status);

-- ---------------------------------------------------------------------------
-- Training provenance: which feedback rows have been used in a training run.
-- ---------------------------------------------------------------------------
ALTER TABLE substitution_results ADD COLUMN IF NOT EXISTS trained_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_sub_results_untrained
    ON substitution_results (trained_at) WHERE trained_at IS NULL;

-- ---------------------------------------------------------------------------
-- MLflow tracking database (separate DB, same server)
-- ---------------------------------------------------------------------------
SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
