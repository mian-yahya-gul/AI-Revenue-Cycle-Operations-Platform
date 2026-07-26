"""Agent 6 — Appeal Generation Agent.

Produces an insurance appeal letter, missing-documentation checklist,
and supporting evidence summary for a denied claim.
"""

from __future__ import annotations

import time

from config.settings import settings
from database.repository import log_agent_action
from models.schemas import (
    AgentLogEntry,
    AppealPackage,
    Claim,
    DenialAnalysis,
    InsuranceFindings,
)
from prompts.agent_prompts import APPEAL_GENERATION_SYSTEM_PROMPT
from tools.appeal_tools import generate_appeal_letter, generate_missing_documentation_checklist
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Appeal Generation Agent"


def _mock_generate(
    claim: Claim, denial_analysis: DenialAnalysis, insurance_findings: InsuranceFindings | None
) -> AppealPackage:
    letter = generate_appeal_letter(claim, denial_analysis, insurance_findings)
    checklist = generate_missing_documentation_checklist(denial_analysis)
    summary = (
        f"Denial category '{denial_analysis.category}' — root cause: {denial_analysis.root_cause} "
        f"Recommended corrections: {'; '.join(denial_analysis.recommended_corrections)}"
    )
    return AppealPackage(
        appeal_letter=letter,
        missing_documentation_checklist=checklist,
        supporting_evidence_summary=summary,
    )


def _llm_generate(
    claim: Claim, denial_analysis: DenialAnalysis, insurance_findings: InsuranceFindings | None
) -> AppealPackage:
    from utils.llm_client import get_structured_llm

    structured_llm = get_structured_llm(AppealPackage)
    policy_context = ""
    if insurance_findings and insurance_findings.policy_references:
        policy_context = "\n".join(
            f"- {p.document} ({p.payer}): {p.excerpt}" for p in insurance_findings.policy_references
        )

    response = structured_llm.invoke(
        [
            {"role": "system", "content": APPEAL_GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Claim ID: {claim.claim_id}\nPatient: {claim.patient_name}\n"
                    f"Payer: {claim.payer.value}\nDate of service: {claim.date_of_service}\n"
                    f"Denial reason code: {denial_analysis.denial_reason_code}\n"
                    f"Root cause: {denial_analysis.root_cause}\n"
                    f"Category: {denial_analysis.category}\n"
                    f"Recommended corrections: {denial_analysis.recommended_corrections}\n\n"
                    f"Relevant policy references:\n{policy_context}"
                ),
            },
        ]
    )
    return response


def run_appeal_generation_agent(
    claim: Claim, denial_analysis: DenialAnalysis, insurance_findings: InsuranceFindings | None
) -> AppealPackage:
    start = time.time()
    logger.info("[%s] Processing claim %s", AGENT_NAME, claim.claim_id)

    if settings.use_mock_llm:
        result = _mock_generate(claim, denial_analysis, insurance_findings)
    else:
        try:
            result = _llm_generate(claim, denial_analysis, insurance_findings)
        except Exception:
            logger.exception("[%s] LLM generation failed, falling back to mock", AGENT_NAME)
            result = _mock_generate(claim, denial_analysis, insurance_findings)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim.claim_id,
            agent_name=AGENT_NAME,
            action="generate_appeal",
            input_summary=f"denial_category={denial_analysis.category}",
            output_summary=f"appeal_id={result.appeal_id}, checklist_items={len(result.missing_documentation_checklist)}",
            duration_ms=duration_ms,
        )
    )
    return result
