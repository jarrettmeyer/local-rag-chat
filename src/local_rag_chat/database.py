import psycopg


class Database:
    """Manages database connections and queries."""

    def __init__(self, host: str, port: str, dbname: str, user: str, password: str):
        self.conn = psycopg.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )

    def insert_document(
        self, doc_id: str, file_name: str, chunks: list, embeddings: list
    ):
        """
        Insert a document, its chunks, and embeddings in a single transaction.
        chunks: list of dicts with keys chunk_id, content, page_number, chunk_number, metadata
        embeddings: list of dicts with keys embedding_id, chunk_id, vector
        """
        import json

        with self.conn as conn:
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

