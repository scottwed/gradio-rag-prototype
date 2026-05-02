import os
from pathlib import Path

import gradio as gr
from openai import OpenAI

from shared.shared import answer

sample_folder = Path(__file__).parent.joinpath("sample").joinpath("gradio_md")
PG_PASS = os.environ.get("PG_PASSWORD", "")
DB_DSN = os.environ.get("DATABASE_URL", f"postgresql://postgres:{PG_PASS}@127.0.0.1:5432/postgres")
ROOT_FOLDER = Path(os.environ.get("MD_ROOT_FOLDER", sample_folder))

EMBED_BASE_URL = "http://127.0.0.1:11434/v1/"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-qwen3-embedding-8b")

CHAT_BASE_URL = "http://127.0.0.1:11434/v1/"
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen3.6-35b-a3b")

my_chat_client = OpenAI(base_url=CHAT_BASE_URL, api_key=os.getenv("CHAT_API_KEY", "local"))
my_embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key=os.getenv("EMBED_API_KEY", "local"))


def gradio_chat(message, history):
    try:
        response = answer(
            query=message,
            history=history,
            chat_client=my_chat_client,
            chat_model=CHAT_MODEL,
            embed_client=my_embed_client,
            embed_model=EMBED_MODEL,
            db_dsn=DB_DSN,
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    # TODO: Make this dynamic, execute only once.
    # load_fresh_db(db_dsn=DB_DSN, root_folder=ROOT_FOLDER)

    demo = gr.ChatInterface(
        fn=gradio_chat,
        title="RAG Prototype",
        description="Ask questions about the processed markdown files.",
    )
    demo.launch()
