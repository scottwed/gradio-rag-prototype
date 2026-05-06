import json
from pathlib import Path

import psycopg
from loguru import logger
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg import Connection, sql

from shared.shared import embed_text
from vector_db.queries import DDL, ROW_COUNT, UPSERT_SQL


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


def iter_text_files(root: Path):
    # TODO: Move these hardcoded items to an input YAML file
    # TODO: Add support for regex patterns
    # TODO: Pre-filter to perform redaction/exclusion of sensitive data
    # TODO: Inspect file header to avoid binaries, PDFs, and images
    # Tweak these lists as needed.  Take care to avoid accidentally returning any files that
    #  could contain sensitive data
    allowed_extensions = ['', '.bat', '.c', '.csv', '.css', '.db', '.h', '.htm', '.html', '.js', '.json',
                          '.lark', '.md', '.ne', '.py', '.results', '.rst', '.sample', '.spatch', '.sh',
                          '.test', '.txt', '.typed', '.xml', '.yaml', '.yml']
    # Note: pyproject.toml might contain secrets, check your source data before adding .toml to allowed extensions.
    blocked_folders = ['.idea', '.git']
    allowed_extensions = [x.lower() for x in allowed_extensions]
    blocked_folders = [x.lower() for x in blocked_folders]

    for path in root.rglob("*"):
        if any(part.lower() in blocked_folders for part in path.parts):
            continue
        if path.is_file():
            if path.suffix.lower() in allowed_extensions and not path.name.startswith('.'):
                yield path
            else:
                logger.warning("Skipping: {}", path.absolute())


def ensure_db(conn: Connection):
    logger.info("Defining empty table, extensions, and indexes if needed")
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def ingest_file(conn: Connection, path: Path, root: Path, embed_client: OpenAI, embed_model: str):
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


def is_table_empty(conn, table_name: str) -> bool:
    result = True
    with conn.cursor() as cur:
        prepared_query = sql.SQL(ROW_COUNT).format(table_name=sql.Identifier(table_name)).as_string(conn)
        row_count = cur.execute(prepared_query).fetchone()[0]
        logger.info("Table {} has {} rows", table_name, row_count)
        result = row_count == 0
    return result


def db_prep(db_dsn: str, root_folder: Path, embed_client: OpenAI, embed_model: str) -> bool:
    with psycopg.connect(db_dsn) as conn:
        ensure_db(conn)
        register_vector(conn)
        if is_table_empty(conn, "documents"):
            logger.info("Starting recursive ingestion of files under: {}", root_folder.absolute())
            if not root_folder.is_dir():
                logger.error("Specified root path must be a folder: {}. Exiting", root_folder.absolute())
                return False
            for md_file in iter_markdown_files(root_folder):
                ingest_file(conn, md_file, root_folder, embed_client, embed_model)
        else:
            logger.info("DB is already populated. Skipping ingestion.")
        conn.commit()
    return True
