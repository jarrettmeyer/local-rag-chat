import pymupdf
import os
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from typing import Any, Dict, List, Tuple


class DocumentProcessor:
    """
    Handles PDF document processing, text extraction, chunking, and embedding
    generation. Extracts text, tables, and image metadata from PDF files and
    converts them into searchable chunks with vector embeddings.
    """

