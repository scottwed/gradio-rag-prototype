from typing import Final, LiteralString

DDL: Final[LiteralString] = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    source_path   text NOT NULL,
    file_name     text NOT NULL,
    relative_path text NOT NULL,
    chunk_index   int NOT NULL DEFAULT 0,
    content       text NOT NULL,
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding     vector(384) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_source_path_chunk_index
    ON documents (source_path, chunk_index);

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS fts tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(file_name, '') || ' ' || coalesce(content, ''))
) STORED;

CREATE INDEX IF NOT EXISTS documents_fts_gin
    ON documents USING gin(fts);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops);
"""

INSERT_SQL: Final[LiteralString] = """
INSERT INTO documents (
    source_path, file_name, relative_path, 
    chunk_index, content, metadata, embedding
)
VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
"""

# Use when the input data contains large, redundant markdown files,
# such as changelogs written by AI coding platforms.
DELETE_DUPLICATES: Final[LiteralString] = """
DELETE FROM documents
WHERE (source_path, chunk_index) IN (
    SELECT source_path, chunk_index
    FROM (
        SELECT 
            source_path, 
            chunk_index,
            COUNT(*) OVER (PARTITION BY content) as occurrence_count,
            ROW_NUMBER() OVER (
                PARTITION BY content 
                ORDER BY LENGTH(source_path) ASC
            ) as rank_in_group
        FROM documents
        WHERE source_path LIKE '%.md'
    ) subquery
    WHERE occurrence_count > 1  -- Limit to duplicate content
      AND rank_in_group > 1     -- Skip the shortest source_path
);
"""

ROW_COUNT: Final[LiteralString] = """
SELECT count(*) as row_count
FROM {table_name}
"""
