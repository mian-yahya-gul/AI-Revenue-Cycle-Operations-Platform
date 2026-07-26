"""Claim and patient lookup tools, exposed as LangChain tools for agent use."""

from __future__ import annotations

from langchain_core.tools import tool

from database import repository
from utils.logger import get_logger

logger = get_logger(__name__)


@tool("claim_lookup_tool")
def claim_lookup_tool(claim_id: str) -> str:
    """Look up a claim by its claim_id and return a summary of its current
    status, payer, billed amount, and workflow stage."""
    claim = repository.get_claim(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."
    return (
        f"Claim {claim.claim_id}: patient={claim.patient_name}, "
        f"payer={claim.payer.value}, status={claim.status.value}, "
        f"risk={claim.risk_level.value}, billed_amount=${claim.billed_amount:.2f}, "
        f"date_of_service={claim.date_of_service}"
    )


@tool("patient_lookup_tool")
def patient_lookup_tool(patient_id: str) -> str:
    """Look up a patient by patient_id and return demographic and payer info."""
    patient = repository.get_patient(patient_id)
    if not patient:
        return f"No patient found with ID {patient_id}."
    return (
        f"Patient {patient.patient_id}: {patient.full_name}, "
        f"DOB={patient.date_of_birth}, payer={patient.payer.value}, "
        f"member_id={patient.member_id}, plan={patient.plan_name}"
    )
