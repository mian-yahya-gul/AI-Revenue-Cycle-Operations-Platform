"""Knowledge Base page — inspect the payer policy corpus backing the RAG
pipeline, and rebuild the vector index on demand."""

from __future__ import annotations

import streamlit as st

from config.settings import PAYER_POLICY_DIR
from rag.loader import load_policy_documents
from rag.vector_store import build_vector_store
from utils.ui_helpers import bootstrap, page_header

st.set_page_config(page_title="Knowledge Base", page_icon="🗂️", layout="wide")
bootstrap()
page_header("Knowledge Base", "The payer policy document corpus powering the Insurance Policy Agent (RAG)")

documents = load_policy_documents()

col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Policy Documents", len(documents))
    payers = sorted({d.metadata["payer"] for d in documents})
    st.write("**Payers covered:**")
    for p in payers:
        st.write(f"- {p}")

    st.write("")
    if st.button("🔄 Rebuild Vector Index", type="primary"):
        with st.spinner("Loading, chunking, embedding, and indexing policy documents..."):
            build_vector_store()
        st.success("Vector index rebuilt.")

with col2:
    st.write(f"**Source directory:** `{PAYER_POLICY_DIR}`")
    for doc in documents:
        with st.expander(f"📄 {doc.metadata['source']} — {doc.metadata['payer']}"):
            st.text(doc.page_content)
