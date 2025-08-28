import os
from typing import Optional


import typer
from dotenv import load_dotenv

from .database import Database
from .document_processor import DocumentProcessor
from .chat_client import ChatClient


load_dotenv()

app = typer.Typer()


def get_env_var(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_database() -> Database:
    db = Database(
        host=get_env_var("POSTGRES_HOST", "localhost"),
        port=get_env_var("POSTGRES_PORT", "5432"),
        dbname=get_env_var("POSTGRES_DB"),
        user=get_env_var("POSTGRES_USER"),
        password=get_env_var("POSTGRES_PASSWORD"),
    )
    return db


@app.command(name="ingest")
def ingest(
    file: str = typer.Argument(),
):
    """Ingest a PDF file and store its chunks and embeddings in the database."""
    db = get_database()
    processor = DocumentProcessor()
    processor.ingest_pdf(file, db)


@app.command(name="chat")
def chat():
    """Start a chat prompt (default command)."""
    db = get_database()
    chat_client = ChatClient()
    chat_client.chat_loop(db)


if __name__ == "__main__":
    app(prog_name="local-rag-chat")
