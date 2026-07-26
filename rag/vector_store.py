"""
Build, persist, and load the vector store for payer policy documents.

Implementation note: this uses a small, dependency-free NumPy cosine-
similarity store rather than FAISS. FAISS ships as a compiled binary
(via SWIG bindings) with wheels that are inconsistently available across
Python versions, OS/architecture combinations, and NumPy ABI versions —
exactly the kind of "works on my machine" friction a portfolio project
should avoid. For a corpus of this size (a handful of payer policy
documents, a few hundred chunks), brute-force cosine similarity over a
NumPy matrix is effectively instant and requires nothing beyond NumPy,
which is already a transitive dependency of the rest of the stack.

The public interface (``build_vector_store`` / ``load_vector_store`` /
``get_vector_store``, and ``similarity_search`` on the returned store)
is intentionally the same shape as the LangChain vector store interface,
so swapping in FAISS (or Chroma, Pinecone, etc.) later only touches this
file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import VECTOR_STORE_DIR
from rag.embeddings import get_embedding_signature, get_embeddings
from rag.loader import load_policy_documents
from rag.splitter import split_documents
from utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_NAME = "payer_policies"


@dataclass
class NumpyVectorStore:
    """A minimal in-memory vector store backed by a NumPy matrix.

    Vectors are L2-normalized at index time, so a dot product against a
    normalized query vector is equivalent to cosine similarity.
    """

    documents: list[Document]
    vectors: np.ndarray  # shape (n_chunks, embedding_dim), L2-normalized
    embeddings_model: Embeddings

    @classmethod
    def from_documents(cls, documents: list[Document], embeddings_model: Embeddings) -> "NumpyVectorStore":
        texts = [doc.page_content for doc in documents]
        raw_vectors = embeddings_model.embed_documents(texts)
        matrix = np.array(raw_vectors, dtype=np.float32)
        matrix = cls._normalize(matrix)
        return cls(documents=documents, vectors=matrix, embeddings_model=embeddings_model)

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if len(self.documents) == 0:
            return []
        query_vector = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_vector) or 1.0
        query_vector = query_vector / query_norm

        scores = self.vectors @ query_vector
        k = min(k, len(self.documents))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [self.documents[i] for i in top_indices]

    def save_local(self, directory, embedding_signature: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{_INDEX_NAME}_vectors.npy", self.vectors)

        serializable_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in self.documents
        ]
        with open(directory / f"{_INDEX_NAME}_documents.json", "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f)

        with open(directory / f"{_INDEX_NAME}_meta.json", "w", encoding="utf-8") as f:
            json.dump({"embedding_signature": embedding_signature, "dimension": self.vectors.shape[1]}, f)

    @classmethod
    def load_local(cls, directory, embeddings_model: Embeddings) -> "NumpyVectorStore":
        vectors = np.load(directory / f"{_INDEX_NAME}_vectors.npy")
        with open(directory / f"{_INDEX_NAME}_documents.json", "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]
        return cls(documents=documents, vectors=vectors, embeddings_model=embeddings_model)


def build_vector_store() -> NumpyVectorStore:
    """Load, chunk, embed, and index all payer policy documents."""
    documents = load_policy_documents()
    if not documents:
        raise RuntimeError(
            "No payer policy documents found. Ensure sample_data/payer_policies "
            "contains .txt files before building the vector store."
        )
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    store = NumpyVectorStore.from_documents(chunks, embeddings)
    store.save_local(VECTOR_STORE_DIR, get_embedding_signature())
    logger.info("Vector store built and persisted to %s (%d chunks)", VECTOR_STORE_DIR, len(chunks))
    return store


def load_vector_store() -> NumpyVectorStore:
    """Load a persisted vector store, building it on first run if absent.

    If the configured embedding provider/model has changed since the index
    was built (e.g. switching from mock mode to a live OPENAI_API_KEY, or
    changing EMBEDDING_MODEL), the persisted vectors have the wrong
    dimensionality for the current embedder. Rather than crash at query
    time with a matmul shape error, detect the mismatch here and
    transparently rebuild the index.
    """
    embeddings = get_embeddings()
    vectors_file = VECTOR_STORE_DIR / f"{_INDEX_NAME}_vectors.npy"
    meta_file = VECTOR_STORE_DIR / f"{_INDEX_NAME}_meta.json"

    if not vectors_file.exists():
        logger.info("No existing vector store found — building a new one")
        return build_vector_store()

    current_signature = get_embedding_signature()
    stored_signature = None
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            stored_signature = json.load(f).get("embedding_signature")

    if stored_signature != current_signature:
        logger.warning(
            "Embedding provider changed (index built with %r, now configured with %r) "
            "— rebuilding the vector index to match.",
            stored_signature, current_signature,
        )
        return build_vector_store()

    return NumpyVectorStore.load_local(VECTOR_STORE_DIR, embeddings)


_store_singleton: NumpyVectorStore | None = None


def get_vector_store() -> NumpyVectorStore:
    """Process-wide cached accessor so Streamlit reruns don't rebuild the index."""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = load_vector_store()
    return _store_singleton
