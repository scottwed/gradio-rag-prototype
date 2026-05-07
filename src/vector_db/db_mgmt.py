from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import psycopg
from loguru import logger
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg import Connection, sql

from shared.shared import embed_text
from vector_db.queries import DDL, ROW_COUNT, UPSERT_SQL


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    # This is primitive, but good enough for the intended content.
    # It's not content aware, meaning that it will accidentally break up
    #  HTML tables and Python methods / class definitions
    # TODO: Incorporate langchain, which has some content aware chunking ability
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
    allowed_extensions = ['', '.bat', '.c', '.conf', '.csv', '.css', '.db', '.h', '.htm', '.html', '.in',
                          '.j2', '.js', '.json',
                          '.lark', '.md', '.ne', '.py', '.results', '.rst', '.sample', '.spatch', '.sh',
                          '.test', '.txt', '.typed', '.xml', '.yaml', '.yml']
    # Note: pyproject.toml might contain secrets, check your source data before adding .toml to allowed extensions.
    blocked_folders = ['.idea', '.git', '.gitlab', '.github']
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


# def ingest_file(conn: Connection, path: Path, root: Path, embed_client: OpenAI, embed_model: str):
#     logger.info("Ingesting  {}", path.absolute())
#     content = path.read_text(encoding="utf-8", errors="ignore")
#     rel = str(path.relative_to(root))
#     chunks = chunk_text(content)
#     rows = []
#     for i, chunk in enumerate(chunks):
#         emb = embed_text(chunk, embed_client=embed_client, embed_model=embed_model)
#         source_type = "text" if not path.suffix else path.suffix
#         meta = {"chunk_index": i, "chunk_count": len(chunks), "source_type": source_type }
#         rows.append((str(path.resolve()), path.name, rel, i, chunk, json.dumps(meta), emb))
#
#     with conn.cursor() as cur:
#         for row in rows:
#             cur.execute(UPSERT_SQL, row)


def is_valid_text_file(path: Path) -> bool:
    """
    Checks if a file is suitable for RAG ingestion.
    Returns False for binaries, executables, images, and DB files.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(2048)
            if not header.strip():
                logger.warning("Skipping empty file {}", path.absolute())
                return False  # Skip empty files

            # Text files rarely contain null bytes; binaries are full of them.
            if b'\x00' in header:
                logger.warning("Skipping suspected binary file (null byte in header) {}", path.absolute())
                return False

            # ELF (Linux Executable), MZ (Windows Executable), PNG, JPEG
            if header.startswith((b'\x7fELF', b'MZ', b'\x89PNG', b'\xff\xd8\xff')):
                logger.warning("Skipping binary or image file {}", path.absolute())
                return False

            # Try UTF-8 decode
            try:
                header.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning("Skipping non utf-8 file {}", path.absolute())
                return False

    except Exception as e:
        logger.exception("Error inspecting {}: {}", path.absolute())
        return False
    return True


def ingest_file_parallel(conn: Connection, path: Path, root: Path, embed_client: OpenAI, embed_model: str):
    logger.info("Ingesting  {}", path.absolute())
    content = path.read_text(encoding="utf-8", errors="ignore")
    if len(content) == 0:
        logger.info("Skipping empty file {}", path.absolute())
        return
    if not is_valid_text_file(path):
        return

    rel = str(path.relative_to(root))
    all_chunks = chunk_text(content)
    rows = []

    batch_size = 32
    chunk_groups = [all_chunks[i:i + batch_size] for i in range(0, len(all_chunks), batch_size)]

    all_vectors = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result_batch in executor.map(lambda g: embed_text(g, embed_client, embed_model), chunk_groups):
            all_vectors.extend(result_batch)

    # logger.info("Vector size was {}", len(all_vectors[0]))
    if len(all_chunks) != len(all_vectors):
        logger.error("Encoding issue: Number of returned vectors ({}) does not "
                     "match number of submitted chunks ({})", len(all_vectors), len(all_chunks))

    for i, emb in enumerate(all_vectors):
        source_type = "text" if not path.suffix else path.suffix
        meta = {"chunk_index": i, "chunk_count": len(all_chunks), "source_type": source_type}
        rows.append((
            str(path.resolve()),    # source_path (pk 1/2)
            path.name,              # file_name
            rel,                    # relative_path inside repo
            i,                      # chunk_index (pk 2/2)
            all_chunks[i],          # content of this chunk
            json.dumps(meta),       # metadata JSONB column
            emb                     # embedding vector (list[float])
        ))

    with conn.cursor() as cur:
        for i, row in enumerate(rows):
            if i % 100 == 0:
                logger.info("Large file upsert progress: {} - {}", row[0], row[-2])
            cur.execute(UPSERT_SQL, row)


def is_table_empty(conn, table_name: str) -> bool:
    result = True
    with conn.cursor() as cur:
        prepared_query = sql.SQL(ROW_COUNT).format(table_name=sql.Identifier(table_name)).as_string(conn)
        row_count = cur.execute(prepared_query).fetchone()[0]
        logger.info("Table {} has {} rows", table_name, row_count)
        result = row_count == 0
    return result


def db_prep(db_dsn: str, input_folders: list[Path], embed_client: OpenAI, embed_model: str) -> bool:
    with psycopg.connect(db_dsn) as conn:
        ensure_db(conn)
        register_vector(conn)
        if is_table_empty(conn, "documents"):
            for input_folder in input_folders:
                if not input_folder.is_dir():
                    logger.error("This import path was not a directory! {}. Aborting", input_folder.absolute())
                    return False

            for input_folder in input_folders:
                logger.info("Starting recursive ingestion of text files under: {}", input_folder.absolute())
                for md_file in iter_text_files(input_folder):
                    ingest_file_parallel(conn, md_file, input_folder, embed_client, embed_model)
        else:
            logger.info("DB is already populated. Skipping ingestion.")
        conn.commit()
    return True
