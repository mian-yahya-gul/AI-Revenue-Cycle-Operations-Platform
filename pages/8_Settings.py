"""Settings page — view active configuration and reset/reseed demo data."""

from __future__ import annotations

import streamlit as st

from config.settings import settings
from database.db import reset_db
from utils.ui_helpers import bootstrap, page_header

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
bootstrap()
page_header("Settings", "Platform configuration and data management")

st.markdown("#### Active Configuration")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**App version:** {settings.app_version}")
    st.write(f"**Mode:** {'Mock (rule-based)' if settings.use_mock_llm else 'Live LLM'}")
    st.write(f"**LLM model:** {settings.llm_model}")
    st.write(f"**Embedding model:** {settings.embedding_model}")
with col2:
    st.write(f"**Validation confidence threshold:** {settings.validation_confidence_threshold}")
    st.write(f"**High risk claim amount:** ${settings.high_risk_claim_amount:,.2f}")
    st.write(f"**Retriever top-k:** {settings.retriever_top_k}")
    st.write(f"**Database path:** `{settings.database_path}`")

st.divider()

st.markdown("#### Data Management")
st.warning("This will permanently delete all claims, patients, appeals, and logs, "
           "and rebuild the database schema from scratch.")

if st.button("⚠️ Reset Database (schema only)"):
    reset_db()
    st.success("Database schema reset. Run the seed script to repopulate demo data:\n\n"
               "`python -m database.seed_data`")

st.divider()
st.markdown("#### About")
st.caption(
    "AI Medical Billing Revenue Cycle Team is a multi-agent AI Operations "
    "Platform demonstrating LangGraph orchestration, RAG, tool calling, "
    "structured outputs, event-driven workflows, and human-in-the-loop "
    "checkpoints for healthcare revenue cycle management."
)
