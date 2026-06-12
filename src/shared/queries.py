from typing import Final, LiteralString

# RETRIEVE: Final[LiteralString] = """
#  WITH q AS (
#      SELECT plainto_tsquery('english', %(query)s) AS tsq,
#             %(emb)s::vector(384) AS qemb
#  )
#  SELECT
#      d.source_path,
#      d.file_name,
#      d.chunk_index,
#      d.content,
#      (0.35 * ts_rank(d.fts, q.tsq) + 0.65 * (1 - (d.embedding <=> q.qemb))) AS score
#  FROM documents d, q
#  ORDER BY score DESC
#  LIMIT %(k)s;
# """


RETRIEVE: Final[LiteralString] = """
WITH fts_ranked AS (
    SELECT 
        source_path, chunk_index,
        ROW_NUMBER() OVER (ORDER BY ts_rank(fts, to_tsquery('english', %(literal_text)s)) DESC) AS fts_rank
    FROM documents
    WHERE fts @@ plainto_tsquery('english', %(literal_text)s)
    LIMIT 100 -- Fetch top candidates to find overlap
),
vector_ranked AS (
    SELECT 
        source_path, chunk_index,
        ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector(384) ASC) AS vector_rank
    FROM documents
    ORDER BY embedding <=> %(emb)s::vector(384) ASC
    LIMIT 100 -- Fetch top candidates to find overlap
)
SELECT 
    d.source_path,
    d.file_name,
    d.chunk_index,
    d.content,
    -- Apply the RRF Formula: 1 / (k + rank)
    COALESCE(1.0 / (60.0 + f.fts_rank), 0.0) + 
    COALESCE(1.0 / (60.0 + v.vector_rank), 0.0) AS rrf_score
FROM documents d
LEFT JOIN fts_ranked f 
  ON f.source_path = d.source_path 
 AND f.chunk_index = d.chunk_index
LEFT JOIN vector_ranked v 
  ON v.source_path = d.source_path 
 AND v.chunk_index = d.chunk_index
WHERE f.chunk_index IS NOT NULL 
   OR v.chunk_index IS NOT NULL
ORDER BY rrf_score DESC
LIMIT %(k)s;
"""