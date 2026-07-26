"""Retrieval interface used by the Insurance Policy Agent and Policy Search page."""

from __future__ import annotations

from langchain_core.documents import Document

from config.settings import settings
from rag.vector_store import get_vector_store
from utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_policy_chunks(
    query: str, payer: str | None = None, k: int | None = None
) -> list[Document]:
    """Retrieve the most relevant policy chunks for a query.

    If ``payer`` is provided, results are filtered to that payer's
    documents post-retrieval so a broader similarity search can still
    surface the right section even when phrasing varies.
    """
    store = get_vector_store()
    top_k = k or settings.retriever_top_k
    search_k = top_k * 4 if payer else top_k

    results = store.similarity_search(query, k=search_k)

    if payer:
        results = [d for d in results if d.metadata.get("payer", "").lower() == payer.lower()]

    results = results[:top_k]
    logger.info(
        "Retrieved %d policy chunks for query=%r payer=%r", len(results), query, payer
    )
    return results


def format_chunks_for_prompt(chunks: list[Document]) -> str:
    if not chunks:
        return "No relevant policy sections found."
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        payer = chunk.metadata.get("payer", "unknown")
        formatted.append(f"[{i}] ({payer} — {source})\n{chunk.page_content.strip()}")
    return "\n\n".join(formatted)
