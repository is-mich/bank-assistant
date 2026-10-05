"""
ingestion/ingestor.py
---------------------
Responsible for: loading files + splitting into chunks + storing in vector DB.
"""

from pathlib import Path
from typing import List, Optional
import logging

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import (
    PyPDFium2Loader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader,
)
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_config
from .structure_splitter import split_by_structure

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """
    Loads a document, splits it into chunks, and stores them in the vector store.
    Accepts any vectorstore compatible with LangChain's add_documents() API.
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def ingest(
        self,
        file_path: str,
        doc_type: Optional[str] = None,
        product_category: Optional[str] = None,
        effective_date: Optional[str] = None,
    ) -> int:
        """
        Full ingestion pipeline: Load → Split → Store.
        Optional metadata is attached to every chunk. A document is a duplicate
        when the same filename (and effective_date, if given) is already stored.
        Returns the number of chunks added.
        """
        config = get_config()
        docs = self._load(file_path)
        if not docs:
            raise ValueError(f"No content extracted from: {file_path}")

        if config.chunking_strategy == "structure":
            chunks = split_by_structure(docs, config.chunk_size, config.chunk_overlap)
        elif config.chunking_strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=["\n\n", "\n", ". ", "; ", " "],
            )
            chunks = splitter.split_documents(docs)
        else:
            raise ValueError(f"Unknown chunking_strategy: {config.chunking_strategy!r}")

        # Normalize source metadata → just filename (not full temp path)
        filename = Path(file_path).name
        # Chroma rejects None metadata values, so only set the ones provided.
        extra = {
            key: value
            for key, value in (
                ("doc_type", doc_type),
                ("product_category", product_category),
                ("effective_date", effective_date),
            )
            if value
        }
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata.update(extra)

        # Duplicate detection — same filename, and same effective_date when given
        where = {"source": filename}
        if effective_date:
            where = {"$and": [where, {"effective_date": effective_date}]}
        existing = self.vectorstore._collection.get(where=where, limit=1)
        if existing["ids"]:
            version = f" ({effective_date})" if effective_date else ""
            raise ValueError(f"**{filename}**{version} is already in the knowledge base.")

        self.vectorstore.add_documents(chunks)
        logger.info(f"Ingested {len(chunks)} chunks from {filename}")
        return len(chunks)

    def _load(self, file_path: str) -> List:
        """Route file to the correct LangChain Document Loader."""
        path = Path(file_path)
        # pypdf drops Greek characters/spaces on some fonts; pdfium preserves them.
        loaders = {
            ".pdf":  PyPDFium2Loader,
            ".txt":  TextLoader,
            ".md":   UnstructuredMarkdownLoader,
            ".docx": Docx2txtLoader,
        }
        loader_cls = loaders.get(path.suffix.lower())
        if not loader_cls:
            raise ValueError(f"Unsupported format: {path.suffix}")
        return loader_cls(str(path)).load()
