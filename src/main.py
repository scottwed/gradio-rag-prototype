import os
from pathlib import Path
from sys import stderr

import gradio as gr
from loguru import logger
from openai import OpenAI

from shared.shared import answer
from vector_db.db_mgmt import db_prep

logger.remove()  # Remove all existing handlers
logger.add(stderr, level="INFO")  # Prevent debug and lower from appearing on console
logger.add("proto_rag.log", rotation="5 MB", retention=10)  # Write detailed logs with rotation

# sample_folder = Path(__file__).parent.parent.joinpath("sample").joinpath("gradio_md")
input_folders = [Path(r'E:\git\lark'), Path(r'E:\git\bind9')]
PG_PASS = os.environ.get("PG_PASSWORD", "")
DB_DSN = os.environ.get("DATABASE_URL", f"postgresql://postgres:{PG_PASS}@127.0.0.1:5432/postgres")

EMBED_BASE_URL = "http://127.0.0.1:11434/v1/"
CHAT_BASE_URL = "http://127.0.0.1:11434/v1/"
# EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-8b")
# CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3.6-35b-a3b")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-granite-embedding-107m-multilingual")
CHAT_MODEL = os.getenv("CHAT_MODEL", "granite-4.1-30b")


my_chat_client = OpenAI(base_url=CHAT_BASE_URL, api_key=os.getenv("CHAT_API_KEY", "local"))
my_embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key=os.getenv("EMBED_API_KEY", "local"))


def gradio_chat(message, history):
    # TODO: Enhance the context that's returned, pulling all chunks from the matched file.
    # TODO: If that's insufficient, convert to a tool calling architecture.
    try:
        response = answer(
            query=message,
            history=history,
            chat_client=my_chat_client,
            chat_model=CHAT_MODEL,
            embed_client=my_embed_client,
            embed_model=EMBED_MODEL,
            db_dsn=DB_DSN,
            chunk_limit=10,
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Dynamic, will ingest only once.
    ready_flag = db_prep(db_dsn=DB_DSN, input_folders=input_folders, embed_client=my_embed_client, embed_model=EMBED_MODEL)
    if ready_flag:
        demo = gr.ChatInterface(
            fn=gradio_chat,
            title="RAG Prototype",
            description="Ask questions about the ingested markdown files.",
        )
        demo.launch()
    else:
        logger.error("Unable to run, database was not viable.")
