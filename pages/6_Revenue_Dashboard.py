"""Revenue Dashboard page — financial and operational analytics across
the full claim portfolio."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import repository
from utils.ui_helpers import bootstrap, page_header

st.set_page_config(page_title="Revenue Dashboard", page_icon="💰", layout="wide")
bootstrap()
page_header("Revenue Dashboard", "Financial and operational analytics across the claim portfolio")

claims = repository.list_claims()
if not claims:
    st.info("No claims yet. Seed sample data or submit a new claim first.")
    st.stop()

rows = [
    {
        "claim_id": c.claim_id,
        "patient_name": c.patient_name,
        "payer": c.payer.value,
        "billed_amount": c.billed_amount,
        "status": c.status.value.replace("_", " ").title(),
        "risk_level": c.risk_level.value,
        "coding_confidence": c.coding_result.overall_confidence if c.coding_result else None,
        "date_of_service": c.date_of_service,
    }
    for c in claims
]
df = pd.DataFrame(rows)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Billed", f"${df['billed_amount'].sum():,.2f}")
col2.metric("Avg. Claim Value", f"${df['billed_amount'].mean():,.2f}")
col3.metric("Denied Claim Value", f"${df.loc[df['status'] == 'Denied', 'billed_amount'].sum():,.2f}")
col4.metric(
    "At-Risk Value (High Risk)",
    f"${df.loc[df['risk_level'] == 'high', 'billed_amount'].sum():,.2f}",
)

left, right = st.columns(2)
with left:
    st.markdown("#### Billed Amount by Payer")
    by_payer = df.groupby("payer")["billed_amount"].sum().sort_values(ascending=False)
    st.bar_chart(by_payer)

with right:
    st.markdown("#### Billed Amount by Status")
    by_status = df.groupby("status")["billed_amount"].sum().sort_values(ascending=False)
    st.bar_chart(by_status)

st.markdown("#### Risk Distribution")
risk_counts = df["risk_level"].value_counts()
st.bar_chart(risk_counts)

st.markdown("#### Full Claim Portfolio")
st.dataframe(
    df.rename(columns={
        "claim_id": "Claim ID", "patient_name": "Patient", "payer": "Payer",
        "billed_amount": "Billed Amount", "status": "Status", "risk_level": "Risk",
        "coding_confidence": "Coding Confidence", "date_of_service": "Date of Service",
    }),
    use_container_width=True,
    hide_index=True,
)
