"""Chunking strategy for payer policy documents."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks sized for policy retrieval.

    Policy documents are structured with headings (e.g. "Coverage Criteria",
    "Prior Authorization Requirements"), so we split on paragraph and
    section boundaries first, falling back to sentence/word boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks
