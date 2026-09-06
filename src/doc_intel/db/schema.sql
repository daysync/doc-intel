-- Applied on startup (idempotent). Stage 3 keeps the schema in one file; a migration tool
-- arrives when the schema starts changing under deployed data.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,
    filename    text NOT NULL,
    mime        text NOT NULL,
    size_bytes  integer NOT NULL,
    status      text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    result      jsonb,
    error       text
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        integer NOT NULL DEFAULT 1,
    kind        text NOT NULL,
    field_path  text,
    text        text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(768),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
-- HNSW approximate index for cosine search; exact scan is fine at this size but the index
-- is what production uses, so the query plans are realistic from the start.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
