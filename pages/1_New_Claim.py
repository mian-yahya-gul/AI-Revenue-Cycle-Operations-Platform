"""New Claim page — submit a claim and watch it move through the
collaborative multi-agent workflow live: Clinical Documentation ->
Coding <-> Validation (feedback loop) -> Insurance Policy (RAG) ->
Compliance (quality gate) -> Revenue Cycle Manager (orchestrator). If
documentation is too sparse to proceed, the workflow pauses here and
this page lets you supply additional notes to resume it."""

from __future__ import annotations

import streamlit as st

from database import repository
from graph.events import resume_claim_with_documentation, submit_new_claim
from models.schemas import Claim, ClaimStatus, Payer
from utils.ui_helpers import bootstrap, page_header, risk_badge, status_badge

st.set_page_config(page_title="New Claim", page_icon="🆕", layout="wide")
bootstrap()
page_header("Submit New Claim", "Runs the collaborative multi-agent claim intake workflow in real time")

patients = repository.list_patients()
if not patients:
    st.warning("No patients found. Run `python -m database.seed_data` first, or this form "
               "will create an ad-hoc patient.")

with st.form("new_claim_form"):
    col1, col2 = st.columns(2)

    with col1:
        if patients:
            patient_options = {f"{p.full_name} ({p.payer.value})": p for p in patients}
            selected_label = st.selectbox("Patient", options=list(patient_options.keys()))
            selected_patient = patient_options[selected_label]
        else:
            selected_patient = None
            first_name = st.text_input("First name", "Jane")
            last_name = st.text_input("Last name", "Doe")
            payer_choice = st.selectbox("Payer", options=[p.value for p in Payer])

        date_of_service = st.date_input("Date of service")
        billed_amount = st.number_input("Billed amount ($)", min_value=0.0, value=500.0, step=25.0)

    with col2:
        physician_notes = st.text_area(
            "Physician notes",
            height=260,
            value=(
                "Chief Complaint: Chest pain\n\n"
                "HPI: Patient presents with acute chest pain radiating to the left arm.\n\n"
                "Assessment: Chest pain, unspecified.\n\n"
                "Plan: EKG obtained, chest x-ray obtained. Basic metabolic panel ordered. "
                "Emergency department level 4 visit.\n\n"
                "Physician: Dr. Sarah Chen\nSigned and dated by attending physician."
            ),
        )

    submitted = st.form_submit_button("Submit Claim & Run Workflow", type="primary")

if submitted:
    if selected_patient is not None:
        patient_id = selected_patient.patient_id
        patient_name = selected_patient.full_name
        payer = selected_patient.payer
    else:
        from database.repository import save_patient
        from models.schemas import Patient

        new_patient = Patient(
            first_name=first_name,
            last_name=last_name,
            date_of_birth="1990-01-01",
            payer=Payer(payer_choice),
            member_id="ADHOC-000000",
        )
        save_patient(new_patient)
        patient_id = new_patient.patient_id
        patient_name = new_patient.full_name
        payer = new_patient.payer

    claim = Claim(
        patient_id=patient_id,
        patient_name=patient_name,
        payer=payer,
        date_of_service=str(date_of_service),
        billed_amount=billed_amount,
        physician_notes_raw=physician_notes,
    )

    with st.status("Running claim intake workflow...", expanded=True) as status_box:
        st.write("🩺 Clinical Documentation Agent extracting diagnoses & procedures...")
        st.write("🧾 Medical Coding Agent assigning ICD-10 / CPT codes...")
        st.write("✅ Claim Validation Agent checking completeness & duplicates "
                 "(may send coding back for revision)...")
        st.write("📚 Insurance Policy Agent retrieving payer policy via RAG...")
        st.write("🛡️ Compliance Agent reviewing every prior agent's output as a quality gate...")
        st.write("📊 Revenue Cycle Manager coordinating the final decision...")
        result = submit_new_claim(claim)
        status_box.update(
            label="Paused — awaiting documentation" if result.status == ClaimStatus.AWAITING_DOCUMENTATION
            else "Workflow complete",
            state="complete",
        )

    st.session_state["last_submitted_claim_id"] = result.claim_id

if st.session_state.get("last_submitted_claim_id"):
    result = repository.get_claim(st.session_state["last_submitted_claim_id"])

    if result.status == ClaimStatus.AWAITING_DOCUMENTATION:
        st.warning(
            f"Claim **{result.claim_id}** is paused — the Clinical Documentation Agent couldn't "
            f"extract any diagnoses or procedures from the notes provided, so the workflow "
            f"stopped here rather than pushing an empty claim through the rest of the team."
        )
        st.caption("Missing: " + ", ".join(result.clinical_summary.missing_information))
        with st.form("resume_documentation_form"):
            supplemental_notes = st.text_area(
                "Additional physician documentation", height=200,
                placeholder="Paste the corrected or supplemental physician notes here...",
            )
            resume_submitted = st.form_submit_button("Resume Workflow With This Documentation", type="primary")
        if resume_submitted and supplemental_notes.strip():
            with st.status("Resuming workflow from where it paused...", expanded=True) as status_box:
                st.write("🩺 Clinical Documentation Agent re-processing with the new notes...")
                st.write("🧾 Medical Coding Agent, ✅ Claim Validation, 📚 Insurance Policy, "
                         "🛡️ Compliance, and 📊 Revenue Cycle Manager all run fresh, since each "
                         "one depends on the corrected clinical summary...")
                resumed = resume_claim_with_documentation(result.claim_id, supplemental_notes)
                status_box.update(label="Resumed", state="complete")
            st.rerun()
        st.stop()

    st.success(f"Claim **{result.claim_id}** processed.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Status**")
        st.markdown(status_badge(result.status), unsafe_allow_html=True)
    with c2:
        st.markdown("**Risk Level**")
        st.markdown(risk_badge(result.risk_level), unsafe_allow_html=True)
    with c3:
        st.markdown("**Billed Amount**")
        st.write(f"${result.billed_amount:,.2f}")

    if result.revision_count:
        st.info(f"🔁 The Coding <-> Validation feedback loop ran **{result.revision_count}** "
                f"revision cycle(s) before this claim proceeded — see the Coding tab for details.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Clinical Summary", "Coding", "Validation", "Insurance Findings", "Compliance", "Manager Assessment"]
    )

    with tab1:
        cs = result.clinical_summary
        if cs:
            st.write(f"**Diagnoses:** {', '.join(cs.diagnoses) or 'None extracted'}")
            st.write(f"**Procedures:** {', '.join(cs.procedures) or 'None extracted'}")
            st.write(f"**Missing information:** {', '.join(cs.missing_information) or 'None'}")
            st.caption(cs.summary_text)

    with tab2:
        cr = result.coding_result
        if cr:
            for code in cr.icd10_codes:
                st.write(f"🩻 **{code.code}** ({code.description}) — confidence {code.confidence:.2f}")
                st.caption(code.reasoning)
            for code in cr.cpt_codes:
                st.write(f"🔧 **{code.code}** ({code.description}) — confidence {code.confidence:.2f}")
                st.caption(code.reasoning)
            st.info(f"Overall confidence: {cr.overall_confidence:.2f}")
            if result.revision_count:
                st.caption(f"Coding notes: {cr.coding_notes}")

    with tab3:
        vr = result.validation_result
        if vr:
            st.write(f"**Valid:** {vr.is_valid} | **Completeness:** {vr.completeness_score:.2f} | "
                      f"**Duplicate detected:** {vr.duplicate_claim_detected}")
            for issue in vr.issues:
                icon = "🔴" if issue.severity == "critical" else "🟡"
                st.write(f"{icon} **{issue.field}**: {issue.issue}")

    with tab4:
        inf = result.insurance_findings
        if inf:
            st.write(f"**Covered:** {inf.is_covered} | **Prior authorization required:** "
                      f"{inf.prior_authorization_required}")
            st.write(f"**Required documents:** {', '.join(inf.required_documents)}")
            if inf.rationale:
                st.caption(f"Rationale: {inf.rationale}")
            if inf.conflicting_rules:
                for c in inf.conflicting_rules:
                    st.warning(f"⚠️ Conflicting policy rules: {c}")
            for ref in inf.policy_references:
                with st.expander(f"📄 {ref.document} ({ref.payer})"):
                    st.write(ref.excerpt)

    with tab5:
        cp = result.compliance_report
        if cp:
            st.write(f"**Compliant:** {cp.is_compliant} | **Human review required:** {cp.human_review_required}")
            st.caption(cp.rationale)
            for finding in cp.findings:
                icon = "🔴" if finding.severity == "critical" else ("🟡" if finding.severity == "warning" else "🔵")
                st.write(f"{icon} **[{finding.area}]** {finding.finding}")
            st.markdown("**Audit trail**")
            for line in cp.audit_trail:
                st.write(f"- {line}")
        else:
            st.caption("Compliance Agent has not reviewed this claim.")

    with tab6:
        ma = result.manager_assessment
        if ma:
            st.markdown(risk_badge(ma.risk_level), unsafe_allow_html=True)
            st.write(f"**Priority score:** {ma.priority_score:.0f}/100")
            st.write(f"**Next step:** `{ma.next_step}`" + (" — **escalated**" if ma.escalated else ""))
            st.write(f"**Recommended action:** {ma.recommended_action}")
            st.caption(ma.rationale)
