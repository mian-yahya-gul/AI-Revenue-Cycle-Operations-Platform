"""AI Medical Billing Revenue Cycle Team — Streamlit entry point."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import settings
from database import repository
from database.db import init_db
from models.schemas import RiskLevel
from utils.ui_helpers import bootstrap, page_header, risk_badge, status_badge

st.set_page_config(
    page_title="AI Medical Billing Revenue Cycle Team",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

bootstrap()

with st.sidebar:
    st.markdown("### 🏥 Revenue Cycle AI")
    st.caption(settings.app_name)
    if settings.use_mock_llm:
        st.info("Running in **mock mode** — no OPENAI_API_KEY detected. Agents use "
                "deterministic rule-based logic so the full workflow still runs offline.")
    else:
        st.success(f"Live LLM mode — model: `{settings.llm_model}`")
    st.divider()
    st.caption("Multi-Agent Team")
    st.caption(
        "1. Clinical Documentation\n"
        "2. Medical Coding\n"
        "3. Claim Validation\n"
        "4. Insurance Policy (RAG)\n"
        "5. Denial Analysis\n"
        "6. Appeal Generation\n"
        "7. Revenue Cycle Manager"
    )

page_header(
    "Revenue Cycle Dashboard",
    "Real-time overview of claims moving through the AI Operations Platform",
)

stats = repository.get_dashboard_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Claims Today", stats["claims_today"])
col2.metric("Pending Claims", stats["pending_claims"])
col3.metric("Denied Claims", stats["denied_claims"])
col4.metric("Appeals Pending", stats["appeals_pending"])

col5, col6, col7 = st.columns(3)
col5.metric("Submission Ready", stats["submission_ready"])
col6.metric("High Risk Claims", stats["high_risk_claims"])
col7.metric("Avg. Coding Confidence", f"{stats['average_confidence']*100:.0f}%")

st.write("")

left, right = st.columns([2, 1])

with left:
    st.markdown("#### Recent Claims")
    claims = repository.list_claims()[:10]
    if not claims:
        st.info("No claims yet. Seed sample data with `python -m database.seed_data`, "
                 "or submit a claim on the **New Claim** page.")
    else:
        for claim in claims:
            c1, c2, c3, c4 = st.columns([2.2, 1.3, 1.1, 1.1])
            with c1:
                st.write(f"**{claim.claim_id}** — {claim.patient_name}")
                st.caption(f"{claim.payer.value} · DOS {claim.date_of_service}")
            with c2:
                st.markdown(status_badge(claim.status), unsafe_allow_html=True)
            with c3:
                st.markdown(risk_badge(claim.risk_level), unsafe_allow_html=True)
            with c4:
                st.write(f"${claim.billed_amount:,.2f}")
            st.divider()

with right:
    st.markdown("#### AI Recommendations")
    high_risk_claims = repository.list_claims(risk_level=RiskLevel.HIGH)
    if not high_risk_claims:
        st.success("No high-risk claims requiring immediate attention.")
    for claim in high_risk_claims[:5]:
        action = claim.manager_assessment.recommended_action if claim.manager_assessment else "Review needed"
        st.warning(f"**{claim.claim_id}** ({claim.patient_name})\n\n{action}")

    st.markdown("#### Claims by Status")
    all_claims = repository.list_claims()
    if all_claims:
        status_counts = pd.Series([c.status.value.replace("_", " ").title() for c in all_claims]).value_counts()
        st.bar_chart(status_counts)
