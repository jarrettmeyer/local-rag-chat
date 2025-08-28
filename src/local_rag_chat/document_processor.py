from tqdm import tqdm
import os
import re

import uuid

import pymupdf

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

    def ingest_pdf(self, file_path: str, db: Database) -> Doc:
        """Ingest a PDF, chunk it, generate embeddings, and save to the database."""
        print(f"\nReading {file_path}...")

        # Use PyMuPDF to open the file.
        pdf_handle = pymupdf.open(file_path)
        doc_id = uuid.uuid4()
        file_name = os.path.basename(file_path)

        # Create a new ChatClient to generate embeddings
        chat_client = ChatClient(db)

        chunk_objs = []
        for page_num in tqdm(
            range(len(pdf_handle)), desc="Processing pages", unit="page"
        ):
            page = pdf_handle[page_num]
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
                    embedding_id=uuid.uuid4(), chunk_id=chunk_id, vector=vector
                )
                chunk_obj = Chunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    page_number=page_num + 1,
                    chunk_number=chunk_num,
                    metadata=metadata,
                    embedding=embedding,
                )
                chunk_objs.append(chunk_obj)

        print("Inserting database records.")
        doc_obj = Doc(doc_id=doc_id, file_name=file_name, chunks=chunk_objs)
        db.insert_document(doc_obj)
        print("Done!")

        pdf_handle.close()

        return doc_obj
