"""Denied Claims page — view denied claims and trigger the denial/appeal
LangGraph workflow: Denial Analysis -> [appealable?] -> Insurance Policy
-> Compliance (quality gate) -> Appeal Generation -> Revenue Cycle
Manager (orchestrator)."""

from __future__ import annotations

import streamlit as st

from database import repository
from graph.events import process_denial
from models.schemas import ClaimStatus
from utils.ui_helpers import bootstrap, page_header, risk_badge, status_badge

st.set_page_config(page_title="Denied Claims", page_icon="🚫", layout="wide")
bootstrap()
page_header("Denied Claims", "Manage denials and generate appeals through the denial/appeal workflow")

tab_manage, tab_simulate = st.tabs(["Manage Denials", "Simulate a New Denial"])

with tab_manage:
    denied = repository.list_claims(status=ClaimStatus.DENIED)
    appealing = repository.list_claims(status=ClaimStatus.APPEALING)
    ready_resub = repository.list_claims(status=ClaimStatus.READY_FOR_RESUBMISSION)
    needs_review = repository.list_claims(status=ClaimStatus.NEEDS_HUMAN_REVIEW)

    all_denial_related = denied + appealing + ready_resub
    if not all_denial_related:
        st.info("No denied claims yet. Use the **Simulate a New Denial** tab to trigger the workflow.")
    else:
        for claim in all_denial_related:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                with c1:
                    st.write(f"**{claim.claim_id}** — {claim.patient_name} ({claim.payer.value})")
                    if claim.denial_analysis:
                        st.caption(f"{claim.denial_analysis.category.replace('_', ' ').title()}: "
                                    f"{claim.denial_analysis.root_cause}")
                with c2:
                    st.markdown(status_badge(claim.status), unsafe_allow_html=True)
                with c3:
                    st.markdown(risk_badge(claim.risk_level), unsafe_allow_html=True)
                with c4:
                    st.write(f"${claim.billed_amount:,.2f}")

                if claim.appeal_package:
                    with st.expander("View appeal letter & checklist"):
                        if claim.compliance_report:
                            st.caption(f"🛡️ Compliance Agent cleared this for appeal "
                                       f"(compliant={claim.compliance_report.is_compliant})")
                        st.write("**Missing documentation checklist:**")
                        for item in claim.appeal_package.missing_documentation_checklist:
                            st.write(f"- {item}")
                        st.text_area(
                            "Appeal letter", value=claim.appeal_package.appeal_letter,
                            height=250, key=f"letter-{claim.claim_id}", label_visibility="collapsed",
                        )

    if needs_review:
        st.markdown("#### Non-Appealable Denials Requiring Human Review")
        for claim in needs_review:
            if claim.denial_analysis:
                st.error(f"**{claim.claim_id}** — {claim.patient_name}: "
                          f"{claim.denial_analysis.root_cause}")

with tab_simulate:
    st.caption("Select an existing claim and simulate a payer denial to trigger the "
               "Denial Analysis -> Insurance Policy -> Compliance -> Appeal Generation -> "
               "Revenue Cycle Manager workflow.")

    eligible_claims = [
        c for c in repository.list_claims()
        if c.status not in (ClaimStatus.DENIED, ClaimStatus.APPEALING, ClaimStatus.READY_FOR_RESUBMISSION)
    ]

    if not eligible_claims:
        st.info("No eligible claims to simulate a denial for. Submit a new claim first.")
    else:
        options = {f"{c.claim_id} — {c.patient_name}": c.claim_id for c in eligible_claims}
        selected = st.selectbox("Claim", options=list(options.keys()))
        claim_id = options[selected]

        denial_reason_options = {
            "CO-197 — Precertification/Authorization Absent": "Claim denied. Reason: CO-197 — Precertification/Authorization Absent.",
            "CO-16 — Claim lacks information needed for adjudication": "Claim denied. Reason: CO-16 — Claim lacks information needed for adjudication.",
            "CO-11 — Diagnosis inconsistent with procedure": "Claim denied. Reason: CO-11 — The diagnosis is inconsistent with the procedure billed.",
            "CO-18 — Duplicate claim/service": "Claim denied. Reason: CO-18 — Duplicate claim/service.",
            "CO-B7 — Missing physician documentation": "Claim denied. Reason: CO-B7 — Provider certification/documentation not on file.",
            "CO-29 — Timely filing limit expired": "Claim denied. Reason: CO-29 — The time limit for filing has expired.",
            "PR-1 — Patient not eligible on date of service": "Claim denied. Reason: PR-1 — Patient was not eligible for coverage on the date of service.",
        }
        reason_label = st.selectbox("Denial reason (from payer EOB)", options=list(denial_reason_options.keys()))

        if st.button("Trigger Denial Workflow", type="primary"):
            with st.status("Running denial/appeal workflow...", expanded=True) as status_box:
                st.write("🔎 Denial Analysis Agent classifying root cause...")
                st.write("📚 Insurance Policy Agent re-checking coverage (RAG)...")
                st.write("🛡️ Compliance Agent reviewing whether there's a sound basis to appeal...")
                st.write("✉️ Appeal Generation Agent drafting appeal letter...")
                st.write("📊 Revenue Cycle Manager finalizing priority...")
                result = process_denial(claim_id, denial_reason_options[reason_label])
                status_box.update(label="Denial workflow complete", state="complete")

            st.success(f"Claim **{result.claim_id}** is now **{result.status.value.replace('_', ' ')}**.")
            if result.appeal_package:
                st.text_area("Generated appeal letter", value=result.appeal_package.appeal_letter, height=300)
