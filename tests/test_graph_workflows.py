"""Integration tests exercising the full LangGraph workflows end-to-end."""

from __future__ import annotations

import os
import sys
import tempfile

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ["DATABASE_PATH"] = _TMP_DB
os.environ["USE_MOCK_LLM"] = "true"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

from database.db import init_db, reset_db  # noqa: E402
from database.repository import list_events, save_patient  # noqa: E402
from graph.events import process_denial, resume_claim_with_documentation, submit_new_claim  # noqa: E402
from models.schemas import Claim, ClaimStatus, Patient, Payer  # noqa: E402
from rag.vector_store import build_vector_store  # noqa: E402

CHEST_PAIN_NOTE = """\
Chief Complaint: Chest pain

HPI: Patient presents with acute chest pain radiating to the left arm.

Assessment: Chest pain, unspecified.

Plan: EKG obtained, chest x-ray obtained. Basic metabolic panel ordered.
Emergency department level 4 visit.

Physician: Dr. Sarah Chen
Date of Service: 2026-06-14
Signed and dated by attending physician.
"""

DIAGNOSIS_ONLY_NOTE = """\
Chief Complaint: Diabetes follow-up

HPI: Patient presents with type 2 diabetes for routine follow up.

Assessment: Type 2 diabetes mellitus, stable.

Physician: Dr. Michael Alvarez
Signed and dated by attending physician.
"""

INCOMPLETE_NOTE = "Patient seen today. Doing okay. Continue current plan."


@pytest.fixture(scope="module", autouse=True)
def _fresh_db_and_index():
    reset_db()
    init_db()
    build_vector_store()
    yield


def _make_patient(payer: Payer = Payer.UNITEDHEALTHCARE) -> Patient:
    patient = Patient(
        first_name="Test", last_name="Patient", date_of_birth="1980-01-01",
        payer=payer, member_id="TST-000001",
    )
    save_patient(patient)
    return patient


def test_well_documented_claim_reaches_ready_for_submission():
    patient = _make_patient(Payer.CIGNA)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-14",
        billed_amount=350.0,
        physician_notes_raw=CHEST_PAIN_NOTE,
    )
    result = submit_new_claim(claim)

    assert result.status == ClaimStatus.READY_FOR_SUBMISSION
    assert result.clinical_summary is not None
    assert result.coding_result is not None
    assert result.validation_result is not None
    assert result.validation_result.is_valid is True
    assert result.insurance_findings is not None
    assert result.manager_assessment is not None
    assert result.compliance_report is not None
    assert result.compliance_report.is_compliant is True
    assert result.manager_assessment.next_step == "submit_claim"
    assert result.revision_count == 0


def test_diagnosis_only_note_triggers_coding_revision_loop():
    """No procedure is documented, so Medical Coding's first pass has no
    CPT code and low confidence — Claim Validation should send it back
    for a revision, which the Compliance Agent then sees resolved."""
    patient = _make_patient(Payer.AETNA)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-15",
        billed_amount=200.0,
        physician_notes_raw=DIAGNOSIS_ONLY_NOTE,
    )
    result = submit_new_claim(claim)

    assert result.revision_count == 1
    assert len(result.coding_result.cpt_codes) == 1  # fallback code applied on revision
    assert result.validation_result.is_valid is True
    assert any("revision" in line.lower() for line in result.audit_trail)


def test_incomplete_note_pauses_for_documentation():
    """A note with nothing extractable at all should pause the whole
    workflow via the LangGraph interrupt rather than pushing an empty
    claim through the rest of the team."""
    patient = _make_patient(Payer.AETNA)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-15",
        billed_amount=200.0,
        physician_notes_raw=INCOMPLETE_NOTE,
    )
    result = submit_new_claim(claim)

    assert result.status == ClaimStatus.AWAITING_DOCUMENTATION
    assert result.coding_result is None  # downstream agents never ran
    assert result.compliance_report is None


def test_resume_with_documentation_completes_the_workflow():
    """After a pause, supplying real documentation should resume the
    graph from exactly where it left off and let it reach a terminal
    state — only one DOCUMENTATION_REQUIRED event and one DOCUMENT_UPDATED
    event should be recorded, even though LangGraph replays the node
    function internally."""
    patient = _make_patient(Payer.BCBS)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-16",
        billed_amount=500.0,
        physician_notes_raw=INCOMPLETE_NOTE,
    )
    paused = submit_new_claim(claim)
    assert paused.status == ClaimStatus.AWAITING_DOCUMENTATION

    resumed = resume_claim_with_documentation(claim.claim_id, CHEST_PAIN_NOTE)

    assert resumed.status == ClaimStatus.READY_FOR_SUBMISSION
    assert resumed.clinical_summary.diagnoses
    assert resumed.coding_result is not None

    events = list_events(claim_id=claim.claim_id)
    assert sum(1 for e in events if e["event_type"] == "documentation_required") == 1
    assert sum(1 for e in events if e["event_type"] == "document_updated") == 1


def test_resume_with_still_insufficient_documentation_pauses_again():
    patient = _make_patient(Payer.BCBS)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-16",
        billed_amount=500.0,
        physician_notes_raw=INCOMPLETE_NOTE,
    )
    submit_new_claim(claim)

    still_bad = resume_claim_with_documentation(claim.claim_id, "Follow up visit. No changes.")
    assert still_bad.status == ClaimStatus.AWAITING_DOCUMENTATION

    resumed = resume_claim_with_documentation(claim.claim_id, CHEST_PAIN_NOTE)
    assert resumed.status == ClaimStatus.READY_FOR_SUBMISSION

    events = list_events(claim_id=claim.claim_id)
    assert sum(1 for e in events if e["event_type"] == "documentation_required") == 1
    assert sum(1 for e in events if e["event_type"] == "document_updated") == 2


def test_high_billed_amount_flagged_high_risk():
    patient = _make_patient(Payer.BCBS)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-16",
        billed_amount=9500.0,
        physician_notes_raw=CHEST_PAIN_NOTE,
    )
    result = submit_new_claim(claim)
    assert result.manager_assessment.priority_score >= 25


def test_prior_auth_required_service_escalates_via_compliance():
    """MRI is one of the prior-authorization service hints — Insurance
    Policy should flag prior_authorization_required, and the Compliance
    Agent should treat that as a blocking finding requiring human review,
    which the Revenue Cycle Manager then honors as authoritative."""
    patient = _make_patient(Payer.UNITEDHEALTHCARE)
    mri_note = (
        "Chief Complaint: Chronic headache\n"
        "Assessment: Migraine, unspecified.\n"
        "Plan: MRI brain ordered to rule out structural cause.\n"
        "Physician: Dr. Sarah Chen\nSigned and dated by attending physician.\n"
    )
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-19",
        billed_amount=2200.0,
        physician_notes_raw=mri_note,
    )
    result = submit_new_claim(claim)

    assert result.insurance_findings.prior_authorization_required is True
    assert result.compliance_report.human_review_required is True
    assert result.status == ClaimStatus.NEEDS_HUMAN_REVIEW
    assert result.manager_assessment.escalated is True
    assert result.manager_assessment.next_step == "escalate_to_human_review"


def test_denial_workflow_produces_appeal_for_appealable_denial():
    patient = _make_patient(Payer.UNITEDHEALTHCARE)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-17",
        billed_amount=1200.0,
        physician_notes_raw=CHEST_PAIN_NOTE,
    )
    submitted = submit_new_claim(claim)

    result = process_denial(
        submitted.claim_id,
        "Claim denied. Reason: CO-16 — Claim lacks information needed for adjudication.",
    )

    assert result.status == ClaimStatus.READY_FOR_RESUBMISSION
    assert result.denial_analysis is not None
    assert result.denial_analysis.category == "missing_documentation"
    assert result.compliance_report is not None
    assert result.appeal_package is not None
    assert "Appeal" in result.appeal_package.appeal_letter


def test_denial_workflow_routes_non_appealable_to_human_review():
    patient = _make_patient(Payer.CIGNA)
    claim = Claim(
        patient_id=patient.patient_id,
        patient_name=patient.full_name,
        payer=patient.payer,
        date_of_service="2026-06-18",
        billed_amount=800.0,
        physician_notes_raw=CHEST_PAIN_NOTE,
    )
    submitted = submit_new_claim(claim)

    result = process_denial(
        submitted.claim_id,
        "Claim denied. Reason: PR-1 — Patient was not eligible for coverage on the date of service.",
    )

    assert result.status == ClaimStatus.NEEDS_HUMAN_REVIEW
    assert result.appeal_package is None
