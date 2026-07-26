"""Agent 7 — Revenue Cycle Manager Agent.

The orchestration and escalation authority for the platform. Every claim
passes through this agent last in both workflows. It does not just report
risk — it treats the Compliance Agent's human_review_required flag as
authoritative and issues a concrete next_step directive that the graph
acts on to finalize claim status. This is the agent's collaborative role
in the multi-agent team: it reads what every other agent decided and
coordinates what happens next, rather than independently re-deciding
things the Compliance Agent already settled.
"""

from __future__ import annotations

import time

from config.settings import settings
from database.repository import log_agent_action
from models.schemas import (
    AgentLogEntry,
    Claim,
    ComplianceReport,
    InsuranceFindings,
    ManagerAssessment,
    RiskLevel,
    ValidationResult,
)
from prompts.agent_prompts import REVENUE_CYCLE_MANAGER_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Revenue Cycle Manager Agent"


def _mock_assess(
    claim: Claim,
    validation_result: ValidationResult | None,
    insurance_findings: InsuranceFindings | None,
    compliance_report: ComplianceReport | None,
) -> ManagerAssessment:
    score = 20.0
    reasons = []

    if claim.billed_amount >= settings.high_risk_claim_amount:
        score += 25
        reasons.append(f"high billed amount (${claim.billed_amount:,.2f})")

    if validation_result and not validation_result.is_valid:
        score += 30
        reasons.append(f"{len(validation_result.issues)} unresolved validation issue(s)")

    if validation_result and validation_result.duplicate_claim_detected:
        score += 15
        reasons.append("possible duplicate claim")

    if insurance_findings and insurance_findings.prior_authorization_required:
        score += 15
        reasons.append("prior authorization required")

    if claim.denial_analysis:
        score += 20
        reasons.append(f"claim was denied ({claim.denial_analysis.category})")

    if compliance_report and not compliance_report.is_compliant:
        critical = sum(1 for f in compliance_report.findings if f.severity == "critical")
        score += 10 * critical
        reasons.append(f"Compliance Agent flagged {critical} critical finding(s)")

    score = min(score, 100.0)

    # The Compliance Agent's human-review determination is authoritative:
    # the manager escalates on its say-so regardless of what the numeric
    # score alone would suggest.
    escalated = bool(compliance_report and compliance_report.human_review_required)

    if escalated:
        risk_level = RiskLevel.HIGH
        next_step = "escalate_to_human_review"
        action = "Escalate to senior billing specialist — Compliance Agent flagged mandatory human review"
    elif score >= 65:
        risk_level = RiskLevel.HIGH
        next_step = "escalate_to_human_review"
        action = "Escalate to senior billing specialist for immediate review"
    elif score >= 35:
        risk_level = RiskLevel.MEDIUM
        next_step = "queue_for_review"
        action = "Queue for standard review before submission"
    else:
        risk_level = RiskLevel.LOW
        next_step = "submit_claim"
        action = "Proceed with standard workflow; no escalation needed"

    if compliance_report and compliance_report.findings:
        reasons.append(compliance_report.rationale)

    rationale = (
        "; ".join(reasons) if reasons else "No elevated risk factors identified in current claim state."
    )

    return ManagerAssessment(
        risk_level=risk_level,
        priority_score=round(score, 1),
        recommended_action=action,
        rationale=rationale,
        next_step=next_step,
        escalated=escalated,
    )


def _llm_assess(
    claim: Claim,
    validation_result: ValidationResult | None,
    insurance_findings: InsuranceFindings | None,
    compliance_report: ComplianceReport | None,
) -> ManagerAssessment:
    from utils.llm_client import get_structured_llm

    structured_llm = get_structured_llm(ManagerAssessment)
    response = structured_llm.invoke(
        [
            {"role": "system", "content": REVENUE_CYCLE_MANAGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Claim ID: {claim.claim_id}\nBilled amount: ${claim.billed_amount}\n"
                    f"Status: {claim.status.value}\n"
                    f"Validation: {validation_result.model_dump_json() if validation_result else 'N/A'}\n"
                    f"Insurance findings: {insurance_findings.model_dump_json() if insurance_findings else 'N/A'}\n"
                    f"Compliance report: {compliance_report.model_dump_json() if compliance_report else 'N/A'}\n"
                    f"Denial analysis present: {claim.denial_analysis is not None}"
                ),
            },
        ]
    )
    # The compliance flag is a hard business rule, not a suggestion the
    # model can override — enforce it even in live mode.
    if compliance_report and compliance_report.human_review_required:
        response.escalated = True
        response.next_step = "escalate_to_human_review"
        response.risk_level = RiskLevel.HIGH
    return response


def run_revenue_cycle_manager_agent(
    claim: Claim,
    validation_result: ValidationResult | None = None,
    insurance_findings: InsuranceFindings | None = None,
    compliance_report: ComplianceReport | None = None,
) -> ManagerAssessment:
    start = time.time()
    logger.info("[%s] Coordinating next steps for claim %s", AGENT_NAME, claim.claim_id)

    if settings.use_mock_llm:
        result = _mock_assess(claim, validation_result, insurance_findings, compliance_report)
    else:
        try:
            result = _llm_assess(claim, validation_result, insurance_findings, compliance_report)
        except Exception:
            logger.exception("[%s] LLM assessment failed, falling back to mock", AGENT_NAME)
            result = _mock_assess(claim, validation_result, insurance_findings, compliance_report)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim.claim_id,
            agent_name=AGENT_NAME,
            action="orchestrate_next_step",
            input_summary=f"billed_amount={claim.billed_amount}, "
            f"compliance_human_review={compliance_report.human_review_required if compliance_report else 'N/A'}",
            output_summary=f"risk={result.risk_level.value}, priority={result.priority_score}, "
            f"next_step={result.next_step}, escalated={result.escalated}",
            duration_ms=duration_ms,
        )
    )
    return result
