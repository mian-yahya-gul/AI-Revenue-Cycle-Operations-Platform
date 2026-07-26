"""
Seed the database and vector store with realistic demonstration
data.

Running this script:
1. Resets and initializes the SQLite schema.
2. Builds the vector index over the sample payer policy documents.
3. Creates sample patients.
4. Submits several sample claims through the real claim-intake LangGraph
   workflow (Clinical Documentation -> Coding <-> Validation feedback
   loop -> Insurance Policy (RAG) -> Compliance -> Revenue Cycle
   Manager), so the resulting data reflects genuine multi-agent
   collaboration rather than hand-authored fixtures. One claim is
   intentionally under-documented to demonstrate the documentation-pause
   interrupt, then resumed with supplemental notes mid-script.
5. Runs claims through the denial/appeal workflow (Denial Analysis ->
   Compliance -> Appeal Generation -> Revenue Cycle Manager) to populate
   the Denied Claims and Policy Search dashboard pages.

Usage:
    python -m database.seed_data
"""

from __future__ import annotations

from config.settings import PHYSICIAN_NOTES_DIR
from database.db import reset_db
from database.repository import save_patient
from graph.events import process_denial, resume_claim_with_documentation, submit_new_claim
from models.schemas import Claim, ClaimStatus, Patient, Payer
from rag.vector_store import build_vector_store
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_note(filename: str) -> str:
    return (PHYSICIAN_NOTES_DIR / filename).read_text(encoding="utf-8")


SUPPLEMENTAL_NOTE_FOR_KESSLER = """\
Addendum to prior note — additional documentation requested.

Chief Complaint: Follow-up visit for hypertension management

HPI: Patient returns for routine blood pressure check. Reports good
medication adherence, no side effects. No new complaints.

Assessment: Essential hypertension, controlled.

Plan: Basic metabolic panel ordered to monitor renal function. Continue
current antihypertensive regimen. Office visit level 3, established
patient.

Physician: Dr. Michael Alvarez
Date of Service: 2026-06-22
Signed and dated by attending physician.
"""

PATIENTS = [
    Patient(
        first_name="James", last_name="Whitfield", date_of_birth="1971-03-22",
        payer=Payer.UNITEDHEALTHCARE, member_id="UHC-88213764", plan_name="Choice Plus PPO",
    ),
    Patient(
        first_name="Maria", last_name="Gonzalez", date_of_birth="1964-11-05",
        payer=Payer.AETNA, member_id="AET-55290187", plan_name="Aetna Select",
    ),
    Patient(
        first_name="Olivia", last_name="Bennett", date_of_birth="1996-07-14",
        payer=Payer.CIGNA, member_id="CIG-70012349", plan_name="Cigna Open Access Plus",
    ),
    Patient(
        first_name="Daniel", last_name="Okafor", date_of_birth="2017-09-02",
        payer=Payer.BCBS, member_id="BCBS-40398211", plan_name="Blue Options PPO",
    ),
    Patient(
        first_name="Priya", last_name="Natarajan", date_of_birth="1990-01-30",
        payer=Payer.CIGNA, member_id="CIG-88213111", plan_name="Cigna LocalPlus",
    ),
    Patient(
        first_name="Robert", last_name="Kessler", date_of_birth="1958-05-19",
        payer=Payer.UNITEDHEALTHCARE, member_id="UHC-33421900", plan_name="Choice Plus PPO",
    ),
    Patient(
        first_name="Angela", last_name="Torres", date_of_birth="1982-04-11",
        payer=Payer.AETNA, member_id="AET-90881122", plan_name="Aetna Select",
    ),
]


def seed() -> None:
    logger.info("Resetting database schema")
    reset_db()

    logger.info("Building vector store from payer policy documents")
    build_vector_store()

    logger.info("Seeding patients")
    for patient in PATIENTS:
        save_patient(patient)

    logger.info("Submitting sample claims through the claim intake workflow")

    claims_to_submit = [
        (PATIENTS[0], "note_chest_pain.txt", "2026-06-14", 1850.00),
        (PATIENTS[1], "note_diabetes_followup.txt", "2026-06-10", 320.00),
        (PATIENTS[2], "note_wrist_fracture.txt", "2026-06-18", 6400.00),
        (PATIENTS[3], "note_asthma_exacerbation.txt", "2026-06-20", 540.00),
        (PATIENTS[4], "note_uti.txt", "2026-06-21", 210.00),
        (PATIENTS[5], "note_incomplete.txt", "2026-06-22", 950.00),
        (PATIENTS[6], "note_diagnosis_only.txt", "2026-06-23", 275.00),
    ]

    submitted_claims: list[Claim] = []
    for patient, note_file, dos, amount in claims_to_submit:
        claim = Claim(
            patient_id=patient.patient_id,
            patient_name=patient.full_name,
            payer=patient.payer,
            date_of_service=dos,
            billed_amount=amount,
            physician_notes_raw=_load_note(note_file),
        )
        result = submit_new_claim(claim)
        submitted_claims.append(result)
        logger.info(
            "Claim %s for %s -> status=%s risk=%s revisions=%d",
            result.claim_id, patient.full_name, result.status.value, result.risk_level.value,
            result.revision_count,
        )

    # Robert Kessler's note ("Patient seen today. Doing okay...") has
    # nothing extractable at all, so the Clinical Documentation Agent
    # paused the workflow via a LangGraph interrupt() rather than pushing
    # an empty claim through the rest of the team — this demonstrates
    # that pause, followed by a human supplying additional documentation
    # and the workflow resuming from exactly where it left off.
    kessler_claim = submitted_claims[5]
    if kessler_claim.status == ClaimStatus.AWAITING_DOCUMENTATION:
        logger.info(
            "Claim %s paused awaiting documentation — supplying supplemental notes to resume",
            kessler_claim.claim_id,
        )
        resumed = resume_claim_with_documentation(kessler_claim.claim_id, SUPPLEMENTAL_NOTE_FOR_KESSLER)
        submitted_claims[5] = resumed
        logger.info(
            "Claim %s resumed -> status=%s risk=%s", resumed.claim_id, resumed.status.value, resumed.risk_level.value,
        )

    # Angela Torres's note documents a diagnosis with no procedure at
    # all, so Medical Coding's first pass produces no CPT code and low
    # confidence — Claim Validation sends it back for a revision before
    # the claim proceeds, which is reflected in revision_count and the
    # audit trail on the resulting claim.
    logger.info(
        "Claim %s (coding revision loop demo) -> revisions=%d",
        submitted_claims[6].claim_id, submitted_claims[6].revision_count,
    )

    logger.info("Running denial/appeal workflow for two claims")

    denial_scenarios = [
        (submitted_claims[2], "Claim denied. Reason: CO-197 — Precertification/Authorization Absent."),
        (submitted_claims[5], "Claim denied. Reason: CO-16 — Claim lacks information needed for adjudication."),
    ]
    for claim, denial_reason in denial_scenarios:
        result = process_denial(claim.claim_id, denial_reason)
        logger.info(
            "Denial processed for claim %s -> status=%s", result.claim_id, result.status.value
        )

    logger.info("Seeding complete.")


if __name__ == "__main__":
    seed()
