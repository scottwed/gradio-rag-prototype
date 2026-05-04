from typing import Final, LiteralString

RETRIEVE: Final[LiteralString] = """
 WITH q AS (
     SELECT plainto_tsquery('english', %(query)s) AS tsq,
            %(emb)s::vector(4000) AS qemb
 )
 SELECT
     d.source_path,
     d.file_name,
     d.chunk_index,
     d.content,
     (0.35 * ts_rank(d.fts, q.tsq) + 0.65 * (1 - (d.embedding <=> q.qemb))) AS score
 FROM documents d, q
 ORDER BY score DESC
 LIMIT %(k)s;
 """