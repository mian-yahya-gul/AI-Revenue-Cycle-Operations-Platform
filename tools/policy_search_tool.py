"""Policy search tool — the RAG interface exposed as a callable tool."""

from __future__ import annotations

from langchain_core.tools import tool

from rag.retriever import format_chunks_for_prompt, retrieve_policy_chunks


@tool("policy_search_tool")
def policy_search_tool(query: str, payer: str = "") -> str:
    """Search payer policy documents (prior authorization rules, coverage
    criteria, documentation requirements) using semantic retrieval over the
    payer policy knowledge base. Optionally filter by payer name
    (e.g. 'Aetna', 'Cigna', 'UnitedHealthcare', 'Blue Cross Blue Shield')."""
    chunks = retrieve_policy_chunks(query, payer=payer or None)
    return format_chunks_for_prompt(chunks)
