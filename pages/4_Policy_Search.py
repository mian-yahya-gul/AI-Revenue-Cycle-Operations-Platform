"""Policy Search page — direct interface into the RAG pipeline over
payer policy documents (independent of any specific claim)."""

from __future__ import annotations

import streamlit as st

from models.schemas import Payer
from rag.retriever import retrieve_policy_chunks
from utils.ui_helpers import bootstrap, page_header

st.set_page_config(page_title="Policy Search", page_icon="📚", layout="wide")
bootstrap()
page_header("Policy Search", "Semantic search over the payer policy knowledge base (RAG)")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Search query",
        placeholder="e.g. prior authorization requirements for MRI",
    )
with col2:
    payer_filter = st.selectbox("Payer filter", options=["All payers"] + [p.value for p in Payer])

top_k = st.slider("Number of results", min_value=1, max_value=8, value=4)

if st.button("Search", type="primary") and query:
    payer = None if payer_filter == "All payers" else payer_filter
    with st.spinner("Retrieving relevant policy chunks..."):
        chunks = retrieve_policy_chunks(query, payer=payer, k=top_k)

    if not chunks:
        st.warning("No relevant policy sections found. Try a broader query or different payer filter.")
    else:
        st.success(f"Found {len(chunks)} relevant section(s).")
        for i, chunk in enumerate(chunks, start=1):
            with st.container(border=True):
                st.markdown(f"**[{i}] {chunk.metadata.get('payer', 'Unknown')} — "
                             f"{chunk.metadata.get('source', 'unknown')}**")
                st.write(chunk.page_content)

st.divider()
st.caption(
    "This search runs the same retrieval pipeline used internally by the "
    "Insurance Policy Agent: document loading → chunking → embeddings → "
    "cosine-similarity search → payer-filtered ranking."
)
