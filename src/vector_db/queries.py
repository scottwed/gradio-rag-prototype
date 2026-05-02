DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path   text NOT NULL UNIQUE,
    file_name     text NOT NULL,
    relative_path text NOT NULL,
    chunk_index   int NOT NULL DEFAULT 0,
    content       text NOT NULL,
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding     halfvec(4000) NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS fts tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(file_name, '') || ' ' || coalesce(content, ''))
) STORED;

CREATE INDEX IF NOT EXISTS documents_fts_gin
    ON documents USING gin(fts);

CREATE INDEX IF NOT EXISTS documents_embedding_ivfflat
    ON documents USING ivfflat (embedding halfvec_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS documents_source_path_idx
    ON documents (source_path);
"""

UPSERT_SQL = """
INSERT INTO documents (
    source_path, file_name, relative_path, chunk_index, content, metadata, embedding
)
VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT (source_path) DO UPDATE SET
    file_name = EXCLUDED.file_name,
    relative_path = EXCLUDED.relative_path,
    chunk_index = EXCLUDED.chunk_index,
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding,
    updated_at = now();
"""
