import json

import psycopg


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

    def insert_document(
        self, doc_id: str, file_name: str, chunks: list, embeddings: list
    ):
        """
        Insert a document, its chunks, and embeddings in a single transaction.
        chunks: list of dicts with keys chunk_id, content, page_number, chunk_number, metadata
        embeddings: list of dicts with keys embedding_id, chunk_id, vector
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
                    (doc_id, file_name),
                )
                # Insert chunks
                for chunk in chunks:
                    cur.execute(
                        """
                        INSERT INTO public.chunks (chunk_id, doc_id, content, page_number, chunk_number, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        (
                            chunk["chunk_id"],
                            doc_id,
                            chunk["content"],
                            chunk["page_number"],
                            chunk["chunk_number"],
                            json.dumps(chunk["metadata"]),
                        ),
                    )
                # Insert embeddings
                for emb in embeddings:
                    cur.execute(
                        """
                        INSERT INTO public.embeddings (embedding_id, chunk_id, vector)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (embedding_id) DO NOTHING
                        """,
                        (emb["embedding_id"], emb["chunk_id"], emb["vector"]),
                    )
