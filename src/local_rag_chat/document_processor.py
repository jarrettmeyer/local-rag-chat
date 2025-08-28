import os
import re

import uuid

import fitz  # PyMuPDF

from .chat_client import ChatClient
from .database import Database, Doc, Chunk, Embedding


class DocumentProcessor:
    """
    Handles PDF document processing, text extraction, chunking, and embedding
    generation. Extracts text, tables, and image metadata from PDF files and
    converts them into searchable chunks with vector embeddings.
    """

    def __init__(self):
        self.chunk_size = 800

    def ingest_pdf(self, file_path: str, db: Database):
        """Ingest a PDF, chunk it, generate embeddings, and save to the database atomically."""
        pdf = fitz.open(file_path)
        doc_id = uuid.uuid4()
        file_name = os.path.basename(file_path)

        chat_client = ChatClient()

        chunk_objs = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()  # type: ignore
            sentences = re.split(r"(?<=[.!?]) +", text)
            chunks = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= self.chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sent
            if current:
                chunks.append(current.strip())

            for chunk_num, chunk_text in enumerate(chunks):
                chunk_id = uuid.uuid4()
                metadata = {"source": file_name, "page": page_num + 1}
                vector = chat_client.get_embedding(chunk_text)
                embedding = Embedding(
                    embedding_id=uuid.uuid4(),
                    chunk_id=chunk_id,
                    vector=vector
                )
                chunk_obj = Chunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    page_number=page_num + 1,
                    chunk_number=chunk_num,
                    metadata=metadata,
                    embedding=embedding
                )
                chunk_objs.append(chunk_obj)

        doc_obj = Doc(
            doc_id=doc_id,
            file_name=file_name,
            chunks=chunk_objs
        )
        db.insert_document(doc_obj)
