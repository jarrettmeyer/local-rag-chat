import os
from typing import Optional

import typer
from dotenv import load_dotenv

from .database import Database
from .document_processor import DocumentProcessor


load_dotenv()

app = typer.Typer()


def get_env_var(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@app.command(name="ingest")
def ingest(
    file: str = typer.Argument(),
):
    """Ingest a PDF file and store its chunks and embeddings in the database."""
    # Initialize database connection from environment
    db = Database(
        host=get_env_var("POSTGRES_HOST"),
        port=get_env_var("POSTGRES_PORT"),
        dbname=get_env_var("POSTGRES_DB"),
        user=get_env_var("POSTGRES_USER"),
        password=get_env_var("POSTGRES_PASSWORD"),
    )
    processor = DocumentProcessor()
    processor.ingest_pdf(file, db)


if __name__ == "__main__":
    app()
