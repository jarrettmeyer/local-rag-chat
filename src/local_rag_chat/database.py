import json
from typing import Any, Dict, List
from uuid import UUID, uuid4

import psycopg
from pydantic import BaseModel, Field


# Pydantic models
class Embedding(BaseModel):
    embedding_id: UUID = Field(default_factory=uuid4)
    chunk_id: UUID
    vector: List[float]


class Chunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    content: str
    page_number: int
    chunk_number: int
    metadata: Dict[str, Any]
    embedding: Embedding


class Doc(BaseModel):
    doc_id: UUID = Field(default_factory=uuid4)
    file_name: str
    chunks: List[Chunk] = Field(default_factory=lambda: [])


class Database:
    """Encapsulates all database operations."""

    def __init__(self, host: str, port: str, dbname: str, user: str, password: str):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password

    def get_relevant_chunks_by_embedding(
        self, embedding: list, user_id: str, top_k: int = 5
    ):
        """Return the top_k most similar chunks to the given embedding, filtered by user permissions if user_id is provided."""
        with self._open_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunks.content
                    FROM chunks
                    INNER JOIN embeddings ON chunks.chunk_id = embeddings.chunk_id
                    INNER JOIN permissions ON chunks.doc_id = permissions.doc_id
                    WHERE permissions.user_id = %s
                    ORDER BY embeddings.vector <#> %s::vector ASC
                    LIMIT %s
                    """,
                    (user_id, embedding, top_k),
                )

                return [row[0] for row in cur.fetchall()]

    def grant_permission(self, user_id: str, doc_id: str):
        """Grant a user access to a document."""
        with self._open_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO permissions (user_id, doc_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (user_id, doc_id),
                )

    def insert_document(self, doc: Doc):
        """
        Insert a document, its chunks, and embeddings in a single transaction.
        """
        with self._open_connection() as conn:
            with conn.cursor() as cur:
                # Insert doc
                cur.execute(
                    """
                    INSERT INTO public.docs (doc_id, file_name)
                    VALUES (%s, %s)
                    ON CONFLICT (doc_id) DO NOTHING
                    """,
                    (str(doc.doc_id), doc.file_name),
                )
                # Insert chunks and embeddings
                for chunk in doc.chunks:
                    cur.execute(
                        """
                        INSERT INTO public.chunks (chunk_id, doc_id, content, page_number, chunk_number, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        (
                            chunk.chunk_id,
                            doc.doc_id,
                            chunk.content,
                            chunk.page_number,
                            chunk.chunk_number,
                            json.dumps(chunk.metadata),
                        ),
                    )
                    emb = chunk.embedding
                    cur.execute(
                        """
                        INSERT INTO public.embeddings (embedding_id, chunk_id, vector)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (embedding_id) DO NOTHING
                        """,
                        (emb.embedding_id, emb.chunk_id, emb.vector),
                    )

    def list_documents(self) -> List["Doc"]:
        """Return a list of Doc objects for all documents."""
        from uuid import UUID

        docs = []
        with self._open_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_id, file_name FROM docs ORDER BY file_name")
                for row in cur.fetchall():
                    docs.append(
                        Doc(doc_id=UUID(str(row[0])), file_name=row[1], chunks=[])
                    )
        return docs

    def purge_document(self, doc_id):
        """Delete a document and all related chunks and embeddings by doc_id."""
        with self._open_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM docs WHERE doc_id = %s
                    """,
                    (str(doc_id),),
                )

    def revoke_permission(self, user_id: str, doc_id: str):
        """Revoke a user's access to a document."""
        with self._open_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM permissions WHERE user_id = %s AND doc_id = %s
                    """,
                    (user_id, doc_id),
                )

    def _open_connection(self):
        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )
