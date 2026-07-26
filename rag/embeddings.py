"""
Embedding provider selection.

Uses OpenAI embeddings when an API key is configured. When running in
mock mode (no API key — e.g. a recruiter cloning the repo without
credentials), falls back to a deterministic hashing-based embedding so
the full RAG pipeline (chunking, vector indexing, similarity search) is
still exercised end-to-end without any network calls.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_HASH_DIM = 384


class DeterministicHashEmbeddings(Embeddings):
    """A lightweight, dependency-free embedding used only in mock mode.

    Not semantically meaningful in the way a trained model is, but it is
    deterministic and keeps token overlap between similar policy text
    reasonably correlated via shingled hashing, which is sufficient to
    demonstrate the retrieval pipeline mechanics in an offline demo.
    """

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * _HASH_DIM
        tokens = text.lower().split()
        for i in range(len(tokens)):
            shingle = " ".join(tokens[i : i + 2])
            digest = hashlib.sha256(shingle.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % _HASH_DIM
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    if settings.use_mock_llm:
        logger.info("Using DeterministicHashEmbeddings (mock mode, no OPENAI_API_KEY set)")
        return DeterministicHashEmbeddings()

    from langchain_openai import OpenAIEmbeddings

    logger.info("Using OpenAIEmbeddings model=%s", settings.embedding_model)
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def get_embedding_signature() -> str:
    """A short string identifying which embedding provider/model/dimension
    is currently configured. Persisted alongside a built vector index and
    compared on load, so switching between mock mode and a live OpenAI key
    (or between embedding models) triggers an automatic rebuild instead of
    a silent dimension-mismatch crash at query time.
    """
    if settings.use_mock_llm:
        return f"mock-hash:{_HASH_DIM}"
    return f"openai:{settings.embedding_model}"
