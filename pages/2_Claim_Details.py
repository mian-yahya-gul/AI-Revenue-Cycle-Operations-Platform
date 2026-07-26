"""Claim Details page — full drill-down view of a single claim's state
across every agent that has touched it, plus its event trail. If a claim
is paused awaiting documentation, this page also lets you supply
additional notes to resume the workflow from where it left off."""

from __future__ import annotations

import streamlit as st

from database import repository
from graph.events import resume_claim_with_documentation
from models.schemas import ClaimStatus
from utils.ui_helpers import bootstrap, page_header, risk_badge, status_badge

st.set_page_config(page_title="Claim Details", page_icon="🔍", layout="wide")
bootstrap()
page_header("Claim Details", "Drill into a single claim's full agent trail")

claims = repository.list_claims()
if not claims:
    st.info("No claims yet. Seed sample data or submit a new claim first.")
    st.stop()

claim_options = {f"{c.claim_id} — {c.patient_name} ({c.status.value})": c.claim_id for c in claims}
selected_label = st.selectbox("Select a claim", options=list(claim_options.keys()))
claim_id = claim_options[selected_label]
claim = repository.get_claim(claim_id)

c1, c2, c3, c4 = st.columns(4)
c1.markdown("**Status**")
c1.markdown(status_badge(claim.status), unsafe_allow_html=True)
c2.markdown("**Risk**")
c2.markdown(risk_badge(claim.risk_level), unsafe_allow_html=True)
c3.metric("Billed Amount", f"${claim.billed_amount:,.2f}")
c4.write(f"**Payer**\n\n{claim.payer.value}")

st.caption(f"Workflow stage: `{claim.workflow_stage}`" + (
    f" · Coding revisions: {claim.revision_count}" if claim.revision_count else ""
))

if claim.status == ClaimStatus.AWAITING_DOCUMENTATION:
    st.warning(
        "This claim is paused — the Clinical Documentation Agent couldn't extract any "
        "diagnoses or procedures from the notes on file. Supply additional documentation "
        "below to resume the workflow from exactly this point."
    )
    if claim.clinical_summary and claim.clinical_summary.missing_information:
        st.caption("Missing: " + ", ".join(claim.clinical_summary.missing_information))
    with st.form(f"resume_form_{claim.claim_id}"):
        supplemental_notes = st.text_area("Additional physician documentation", height=200)
        resume_clicked = st.form_submit_button("Resume Workflow", type="primary")
    if resume_clicked and supplemental_notes.strip():
        with st.spinner("Resuming workflow — Clinical Documentation re-processes, and every "
                         "downstream agent runs fresh since each depends on the corrected summary..."):
            resume_claim_with_documentation(claim.claim_id, supplemental_notes)
        st.rerun()

st.divider()

with st.expander("📋 Raw Physician Notes", expanded=False):
    st.text(claim.physician_notes_raw)

tabs = st.tabs([
    "Clinical Summary", "Coding", "Validation", "Insurance Findings", "Compliance",
    "Denial Analysis", "Appeal", "Manager Assessment", "Event Trail",
])

with tabs[0]:
    cs = claim.clinical_summary
    if cs:
        st.write(f"**Diagnoses:** {', '.join(cs.diagnoses) or 'None'}")
        st.write(f"**Procedures:** {', '.join(cs.procedures) or 'None'}")
        st.write(f"**Chief complaint:** {cs.chief_complaint or 'N/A'}")
        st.write(f"**Physician:** {cs.physician_name or 'N/A'}")
        st.write(f"**Missing information:** {', '.join(cs.missing_information) or 'None'}")
        st.progress(cs.confidence, text=f"Extraction confidence: {cs.confidence:.2f}")
    else:
        st.caption("Not yet processed by the Clinical Documentation Agent.")

with tabs[1]:
    cr = claim.coding_result
    if cr:
        if claim.revision_count:
            st.info(f"🔁 Sent back for revision {claim.revision_count} time(s) by the "
                    f"Claim Validation Agent before being accepted.")
        st.write("**ICD-10 Codes**")
        for code in cr.icd10_codes:
            st.write(f"- `{code.code}` {code.description} (confidence {code.confidence:.2f})")
            st.caption(code.reasoning)
        st.write("**CPT Codes**")
        for code in cr.cpt_codes:
            st.write(f"- `{code.code}` {code.description} (confidence {code.confidence:.2f})")
            st.caption(code.reasoning)
        st.progress(cr.overall_confidence, text=f"Overall confidence: {cr.overall_confidence:.2f}")
        st.caption(cr.coding_notes)
    else:
        st.caption("Not yet processed by the Medical Coding Agent.")

with tabs[2]:
    vr = claim.validation_result
    if vr:
        st.write(f"**Valid:** {vr.is_valid}  |  **Completeness score:** {vr.completeness_score:.2f}  |  "
                  f"**Duplicate detected:** {vr.duplicate_claim_detected}")
        for issue in vr.issues:
            icon = "🔴" if issue.severity == "critical" else ("🟡" if issue.severity == "warning" else "🔵")
            st.write(f"{icon} [{issue.severity}] **{issue.field}**: {issue.issue}")
    else:
        st.caption("Not yet processed by the Claim Validation Agent.")

with tabs[3]:
    inf = claim.insurance_findings
    if inf:
        st.write(f"**Covered:** {inf.is_covered}  |  **Prior authorization required:** "
                  f"{inf.prior_authorization_required}")
        st.write(f"**Required documents:** {', '.join(inf.required_documents) or 'None'}")
        if inf.rationale:
            st.write(f"**Rationale:** {inf.rationale}")
        if inf.conflicting_rules:
            for c in inf.conflicting_rules:
                st.warning(f"⚠️ {c}")
        st.caption(inf.notes)
        for ref in inf.policy_references:
            with st.expander(f"📄 {ref.document} ({ref.payer})"):
                st.write(ref.excerpt)
    else:
        st.caption("Not yet processed by the Insurance Policy Agent.")

with tabs[4]:
    cp = claim.compliance_report
    if cp:
        st.write(f"**Compliant:** {cp.is_compliant}  |  **Human review required:** {cp.human_review_required}")
        st.caption(cp.rationale)
        if cp.findings:
            st.write("**Findings**")
            for finding in cp.findings:
                icon = "🔴" if finding.severity == "critical" else ("🟡" if finding.severity == "warning" else "🔵")
                st.write(f"{icon} **[{finding.area}]** {finding.finding}")
        st.write("**Audit trail** (what every prior agent contributed)")
        for line in cp.audit_trail:
            st.write(f"- {line}")
    else:
        st.caption("Not yet reviewed by the Compliance Agent.")

with tabs[5]:
    da = claim.denial_analysis
    if da:
        st.write(f"**Denial reason code:** {da.denial_reason_code}")
        st.write(f"**Category:** {da.category}")
        st.write(f"**Appealable:** {da.is_appealable}")
        st.write("**Root cause:**")
        st.write(da.root_cause)
        st.write("**Recommended corrections:**")
        for c in da.recommended_corrections:
            st.write(f"- {c}")
    else:
        st.caption("No denial has been recorded for this claim.")

with tabs[6]:
    ap = claim.appeal_package
    if ap:
        st.write(f"**Appeal ID:** {ap.appeal_id}  |  **Status:** {ap.status}")
        st.write("**Missing documentation checklist:**")
        for item in ap.missing_documentation_checklist:
            st.checkbox(item, key=f"chk-{ap.appeal_id}-{item}", value=False)
        st.write("**Appeal letter:**")
        st.text_area("Appeal Letter", value=ap.appeal_letter, height=350, label_visibility="collapsed")
    else:
        st.caption("No appeal has been generated for this claim.")

with tabs[7]:
    ma = claim.manager_assessment
    if ma:
        st.markdown(risk_badge(ma.risk_level), unsafe_allow_html=True)
        st.write(f"**Priority score:** {ma.priority_score:.0f}/100")
        st.write(f"**Next step:** `{ma.next_step}`" + (" — **escalated**" if ma.escalated else ""))
        st.write(f"**Recommended action:** {ma.recommended_action}")
        st.caption(ma.rationale)
    else:
        st.caption("Not yet assessed by the Revenue Cycle Manager Agent.")

with tabs[8]:
    events = repository.list_events(claim_id=claim.claim_id)
    logs = repository.list_agent_logs(claim_id=claim.claim_id)
    st.write("**Events**")
    for e in events:
        st.write(f"🔔 `{e['created_at']}` — {e['event_type'].replace('_', ' ').title()}")
    st.write("**Agent Activity Log**")
    for log in logs:
        st.write(f"🤖 `{log['created_at']}` — **{log['agent_name']}**: {log['action']} "
                  f"({log['duration_ms']} ms)")
        st.caption(f"Output: {log['output_summary']}")
