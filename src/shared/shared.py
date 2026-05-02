import numpy as np
import psycopg
from loguru import logger
from openai import OpenAI
from pgvector.psycopg import register_vector


def embed_text(text: str, embed_client: OpenAI, embed_model: str) -> list[float]:
    resp = embed_client.embeddings.create(model=embed_model, input=text)
    # Truncate to 4000 to fit pgvector HNSW/IVFFlat limits
    return np.array(resp.data[0].embedding)[:4000].tolist()


def retrieve(
    query,
    embed_client: OpenAI,
    embed_model: str,
    db_dsn: str,
    k=5,
):
    q_emb = embed_text(query, embed_client, embed_model)
    sql = """
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
    with psycopg.connect(db_dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "emb": q_emb, "k": k})
            return cur.fetchall()


def answer(query, history, chat_client: OpenAI, chat_model: str, embed_client: OpenAI, embed_model: str, db_dsn: str):

    try:
        rows = retrieve(query, embed_client=embed_client, embed_model=embed_model, db_dsn=db_dsn, k=6)
        context = "\n\n".join(f"[{r[0]}#chunk{r[2]}]\n{r[3]}" for r in rows)

        messages = [
            {
                "role": "system",
                "content": "Answer only from the provided context. If the context is insufficient, say so.",
            }
        ]
        messages.extend(history)

        messages.append({"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"})

        resp = chat_client.chat.completions.create(model=chat_model, messages=messages, temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        logger.exception(e)
        return "Error during request"
