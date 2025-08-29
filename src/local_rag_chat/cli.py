import os
from typing import Optional

from dotenv import load_dotenv
from typer import Argument, Option, Typer

from .chat_client import ChatClient
from .database import Database
from .document_processor import DocumentProcessor

load_dotenv()

# Create a CLI application with Typer.
app = Typer()


def _get_env_var(name: str, default: Optional[str] = None) -> str:
    """Get an environment variable with optional default."""
    value = os.getenv(name)
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_database() -> Database:
    """Get a Database instance."""
    db = Database(
        host=_get_env_var("POSTGRES_HOST", "localhost"),
        port=_get_env_var("POSTGRES_PORT", "5432"),
        dbname=_get_env_var("POSTGRES_DB"),
        user=_get_env_var("POSTGRES_USER"),
        password=_get_env_var("POSTGRES_PASSWORD"),
    )
    return db


@app.command(name="chat")
def chat(user: str = Option(None, help="User ID for permissions filtering")):
    """Start a chat prompt (default command). Optionally filter by user permissions."""

    # If user is not set, pull from environment
    if not user:
        user = _get_env_var("USER", None)

    db = _get_database()
    chat_client = ChatClient(db)
    chat_client.chat_loop(user)


@app.command(name="grant")
def grant(
    user_id: str = Argument(..., help="User ID to grant permission to"),
    doc_id: str = Argument(..., help="Document ID to grant access for"),
):
    """Grant a user access to a document."""
    db = _get_database()
    db.grant_permission(user_id, doc_id)

    print(f"\nGranted user {user_id} access to document {doc_id}.", end="\n\n")


@app.command(name="ingest")
def ingest_document(file: str = Argument(..., help="File to be ingested")):
    """Ingest a PDF file and store its chunks and embeddings in the database."""
    db = _get_database()
    processor = DocumentProcessor()
    doc = processor.ingest_pdf(file, db)

    print(f"\nIngested document {doc.file_name} ({str(doc.doc_id)}).", end="\n\n")


@app.command(name="list")
def list_documents():
    """List all documents and their IDs."""
    db = _get_database()
    docs = db.list_documents()

    print("\nDoc ID                                  File Name")
    print("-" * 80)

    for doc in docs:
        print(f"{doc.doc_id}    {doc.file_name}")

    print(f"\nFound {len(docs)} documents.", end="\n\n")


@app.command(name="purge")
def purge_document(doc_id: str = Argument(..., help="Document ID to purge")):
    """Delete a document and all related data from the database."""
    db = _get_database()
    db.purge_document(doc_id)

    print(f"\nPurged document {doc_id} and all related data.", end="\n\n")


@app.command(name="revoke")
def revoke(
    user_id: str = Argument(..., help="User ID to revoke permission from"),
    doc_id: str = Argument(..., help="Document ID to revoke access for"),
):
    """Revoke a user's access to a document."""
    db = _get_database()
    db.revoke_permission(user_id, doc_id)
    print(f"Revoked user {user_id}'s access to document {doc_id}.")


if __name__ == "__main__":
    app(prog_name="local-rag-chat")
