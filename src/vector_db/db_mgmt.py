import json
from pathlib import Path

import psycopg
from loguru import logger
from openai import OpenAI
from pgvector.psycopg import register_vector

from shared.shared import embed_text
from vector_db.queries import DDL, UPSERT_SQL


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n\n", start, end)
            if cut == -1 or cut <= start:
                cut = text.rfind("\n", start, end)
            if cut == -1 or cut <= start:
                cut = end
            end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def iter_markdown_files(root: Path):
    for path in root.rglob("*.md"):
        if path.is_file():
            yield path


def ensure_db(conn):
    logger.info("Establishing DB connection and defining empty table if needed")
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def ingest_file(conn, path: Path, root: Path, embed_client: OpenAI, embed_model: str):
    logger.info("Starting ingestion from {}", path.absolute())
    content = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(root))
    chunks = chunk_text(content)

    rows = []
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk, embed_client=embed_client, embed_model=embed_model)
        meta = {"chunk_index": i, "chunk_count": len(chunks), "source_type": "markdown"}
        rows.append((str(path.resolve()), path.name, rel, i, chunk, json.dumps(meta), emb))

    with conn.cursor() as cur:
        for row in rows:
            cur.execute(UPSERT_SQL, row)


def load_fresh_db(db_dsn: str, root_folder: Path, embed_client: OpenAI, embed_model: str):
    with psycopg.connect(db_dsn) as conn:
        ensure_db(conn)
        register_vector(conn)
        logger.info("Starting recursive ingestion of files under: {}", root_folder.absolute())
        if not root_folder.is_dir():
            logger.error("Specified root path must be a folder: {}.  Exiting", root_folder.absolute())
        for md_file in iter_markdown_files(root_folder):
            ingest_file(conn, md_file, root_folder, embed_client, embed_model)
        conn.commit()
