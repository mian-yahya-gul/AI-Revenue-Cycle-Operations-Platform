"""Agent 5 — Denial Analysis Agent.

Given a denied claim and the payer's denial reason, determines root
cause, classification, and recommended corrections.
"""

from __future__ import annotations

import time

from config.settings import settings
from database.repository import log_agent_action
from models.schemas import AgentLogEntry, Claim, DenialAnalysis
from prompts.agent_prompts import DENIAL_ANALYSIS_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Denial Analysis Agent"

# Maps common payer denial reason codes to category + recommended corrections.
_DENIAL_CODE_MAP: dict[str, dict] = {
    "CO-197": {
        "category": "prior_auth",
        "root_cause": "The service required prior authorization/precertification, "
        "which was not on file at the time of claim submission.",
        "corrections": [
            "Obtain a retroactive/retrospective authorization request from the payer",
            "Submit clinical documentation supporting medical necessity with the retro request",
            "Include a written explanation for why authorization was not obtained in advance",
        ],
    },
    "CO-16": {
        "category": "missing_documentation",
        "root_cause": "The claim was missing one or more required data elements "
        "(e.g. diagnosis code, provider NPI, or date of service).",
        "corrections": [
            "Review the claim for missing required fields",
            "Resubmit a corrected claim with all required elements populated",
        ],
    },
    "CO-11": {
        "category": "coding_error",
        "root_cause": "The billed procedure code is inconsistent with the "
        "submitted diagnosis code, suggesting a coding mismatch.",
        "corrections": [
            "Review clinical documentation and confirm the correct ICD-10/CPT pairing",
            "Resubmit a corrected claim with an updated diagnosis or procedure code",
        ],
    },
    "CO-18": {
        "category": "duplicate",
        "root_cause": "The claim was flagged as a duplicate of a previously "
        "submitted claim for the same patient, date of service, and procedure.",
        "corrections": [
            "Confirm whether this is truly a distinct, medically necessary repeat service",
            "If distinct, resubmit with the appropriate repeat-procedure modifier (-76, -77, or -91) "
            "and supporting documentation",
        ],
    },
    "CO-B7": {
        "category": "missing_documentation",
        "root_cause": "Supporting physician documentation (notes or operative report) "
        "could not be produced to support the billed service.",
        "corrections": [
            "Locate and attach the signed physician note or operative report",
            "Resubmit the claim with the supporting documentation included",
        ],
    },
    "CO-29": {
        "category": "other",
        "root_cause": "The claim was submitted after the payer's timely filing limit.",
        "corrections": [
            "Verify the original date of service and submission date",
            "If a timely-filing exception applies (e.g. eligibility delay), submit proof with the appeal",
        ],
    },
    "PR-1": {
        "category": "eligibility",
        "root_cause": "The patient was not eligible for coverage on the date of service.",
        "corrections": [
            "Verify patient eligibility and coverage effective dates with the payer",
            "If eligibility was active, submit eligibility verification with the appeal",
        ],
    },
}

_DEFAULT_DENIAL = {
    "category": "other",
    "root_cause": "The denial reason code did not map to a known category; "
    "manual review of the payer's explanation of benefits is required.",
    "corrections": ["Review the full remittance advice for payer-specific denial detail"],
}


def _mock_analyze(claim: Claim, denial_reason: str) -> DenialAnalysis:
    code_match = None
    for code in _DENIAL_CODE_MAP:
        if code.lower() in denial_reason.lower():
            code_match = code
            break

    info = _DENIAL_CODE_MAP.get(code_match, _DEFAULT_DENIAL)
    reason_code = code_match or "UNSPECIFIED"

    return DenialAnalysis(
        denial_reason_code=reason_code,
        root_cause=info["root_cause"],
        category=info["category"],
        recommended_corrections=info["corrections"],
        is_appealable=info["category"] != "eligibility",
    )


def _llm_analyze(claim: Claim, denial_reason: str) -> DenialAnalysis:
    from utils.llm_client import get_structured_llm

    structured_llm = get_structured_llm(DenialAnalysis)
    response = structured_llm.invoke(
        [
            {"role": "system", "content": DENIAL_ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Claim ID: {claim.claim_id}\nPayer: {claim.payer.value}\n"
                    f"Billed amount: ${claim.billed_amount}\n"
                    f"Denial reason from payer: {denial_reason}"
                ),
            },
        ]
    )
    return response


def run_denial_analysis_agent(claim: Claim, denial_reason: str) -> DenialAnalysis:
    start = time.time()
    logger.info("[%s] Processing claim %s", AGENT_NAME, claim.claim_id)

    if settings.use_mock_llm:
        result = _mock_analyze(claim, denial_reason)
    else:
        try:
            result = _llm_analyze(claim, denial_reason)
        except Exception:
            logger.exception("[%s] LLM analysis failed, falling back to mock", AGENT_NAME)
            result = _mock_analyze(claim, denial_reason)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim.claim_id,
            agent_name=AGENT_NAME,
            action="analyze_denial",
            input_summary=denial_reason[:200],
            output_summary=f"category={result.category}, appealable={result.is_appealable}",
            duration_ms=duration_ms,
        )
    )
    return result
