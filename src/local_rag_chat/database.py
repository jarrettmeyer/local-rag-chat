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
    def __init__(self, host: str, port: str, dbname: str, user: str, password: str):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password

    def open_connection(self):
        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )

    def get_relevant_chunks_by_embedding(self, embedding: list, top_k: int = 5):
        """Return the top_k most similar chunks to the given embedding."""
        with self.open_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content FROM chunks
                    JOIN embeddings ON chunks.chunk_id = embeddings.chunk_id
                    ORDER BY embeddings.vector <#> %s::vector ASC
                    LIMIT %s
                    """,
                    (embedding, top_k),
                )
                return [row[0] for row in cur.fetchall()]

    def insert_document(self, doc: Doc):
        """
        Insert a document, its chunks, and embeddings in a single transaction.
        """
        with self.open_connection() as conn:
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
