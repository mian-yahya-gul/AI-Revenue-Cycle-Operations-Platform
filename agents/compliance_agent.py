"""Compliance Agent — the quality gate every claim passes through before
it can be marked ready for submission or resubmission.

Unlike the other agents, this one does not generate new clinical, coding,
or policy findings. It *reviews* what the Clinical Documentation, Medical
Coding, Claim Validation, and Insurance Policy agents have already put
into shared state, cross-checks them against each other, and produces a
single authoritative compliance determination — including whether
mandatory human review is required. The Revenue Cycle Manager acts on
this determination directly rather than re-deriving it.
"""

from __future__ import annotations

import time

from config.settings import settings
from database.repository import log_agent_action
from models.schemas import (
    AgentLogEntry,
    Claim,
    ClinicalSummary,
    CodingResult,
    ComplianceFinding,
    ComplianceReport,
    InsuranceFindings,
    ValidationResult,
)
from prompts.agent_prompts import COMPLIANCE_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Compliance Agent"


def _build_audit_trail(
    claim: Claim,
    clinical_summary: ClinicalSummary | None,
    coding_result: CodingResult | None,
    validation_result: ValidationResult | None,
    insurance_findings: InsuranceFindings | None,
    prior_logs: list[str],
) -> list[str]:
    """Assemble a human-readable audit trail from every upstream agent's
    contribution, in execution order. This is what a human reviewer or
    auditor reads to understand how the platform arrived at its decision,
    without having to inspect raw agent output."""
    trail = list(prior_logs)
    if claim.revision_count:
        trail.append(
            f"Coding <-> Validation feedback loop ran {claim.revision_count} revision "
            f"cycle(s) before proceeding."
        )
    return trail


def _mock_review(
    claim: Claim,
    clinical_summary: ClinicalSummary | None,
    coding_result: CodingResult | None,
    validation_result: ValidationResult | None,
    insurance_findings: InsuranceFindings | None,
    prior_logs: list[str],
) -> ComplianceReport:
    findings: list[ComplianceFinding] = []
    human_review_required = False

    # --- Documentation completeness -----------------------------------
    if clinical_summary and clinical_summary.missing_information:
        severity = "critical" if clinical_summary.confidence < 0.5 else "warning"
        findings.append(
            ComplianceFinding(
                area="documentation",
                finding=f"Clinical Documentation Agent flagged {len(clinical_summary.missing_information)} "
                f"missing item(s): {', '.join(clinical_summary.missing_information)}",
                severity=severity,
            )
        )
        if severity == "critical":
            human_review_required = True

    # --- Coding adequacy -------------------------------------------------
    if coding_result:
        if not coding_result.icd10_codes or not coding_result.cpt_codes:
            findings.append(
                ComplianceFinding(
                    area="coding",
                    finding="Claim is missing required ICD-10 or CPT codes after the "
                    "coding revision loop completed.",
                    severity="critical",
                )
            )
            human_review_required = True
        elif coding_result.overall_confidence < settings.validation_confidence_threshold:
            findings.append(
                ComplianceFinding(
                    area="coding",
                    finding=f"Coding confidence ({coding_result.overall_confidence:.2f}) remains below "
                    f"the {settings.validation_confidence_threshold} threshold after revision.",
                    severity="warning",
                )
            )

    # --- Validation results ------------------------------------------------
    if validation_result:
        critical_issues = [i for i in validation_result.issues if i.severity == "critical"]
        if critical_issues:
            findings.append(
                ComplianceFinding(
                    area="validation",
                    finding=f"{len(critical_issues)} unresolved critical validation issue(s) remain.",
                    severity="critical",
                )
            )
            human_review_required = True
        if validation_result.duplicate_claim_detected:
            findings.append(
                ComplianceFinding(
                    area="validation",
                    finding="Possible duplicate claim detected.",
                    severity="warning",
                )
            )

    # --- Payer compliance -----------------------------------------------
    if insurance_findings:
        if insurance_findings.prior_authorization_required:
            findings.append(
                ComplianceFinding(
                    area="prior_authorization",
                    finding="Payer policy requires prior authorization, and no authorization "
                    "reference is on file for this claim. Submission must be blocked until "
                    "authorization is obtained.",
                    severity="critical",
                )
            )
            human_review_required = True
        if not insurance_findings.is_covered:
            findings.append(
                ComplianceFinding(
                    area="coverage",
                    finding="Insurance Policy Agent determined this service is not covered "
                    "under the current payer policy.",
                    severity="critical",
                )
            )
            human_review_required = True
        if insurance_findings.conflicting_rules:
            findings.append(
                ComplianceFinding(
                    area="policy_conflict",
                    finding=f"Conflicting policy rules found: {'; '.join(insurance_findings.conflicting_rules)}",
                    severity="warning",
                )
            )

    is_compliant = not any(f.severity == "critical" for f in findings)
    audit_trail = _build_audit_trail(
        claim, clinical_summary, coding_result, validation_result, insurance_findings, prior_logs
    )

    if not findings:
        rationale = "No compliance issues found across documentation, coding, validation, or payer policy."
    else:
        critical_count = sum(1 for f in findings if f.severity == "critical")
        rationale = (
            f"{critical_count} critical and {len(findings) - critical_count} non-critical "
            f"finding(s) across upstream agents. "
            f"{'Mandatory human review required.' if human_review_required else 'No mandatory human review triggered.'}"
        )

    return ComplianceReport(
        is_compliant=is_compliant,
        findings=findings,
        human_review_required=human_review_required,
        audit_trail=audit_trail,
        rationale=rationale,
    )


def _llm_review(
    claim: Claim,
    clinical_summary: ClinicalSummary | None,
    coding_result: CodingResult | None,
    validation_result: ValidationResult | None,
    insurance_findings: InsuranceFindings | None,
    prior_logs: list[str],
) -> ComplianceReport:
    from utils.llm_client import get_structured_llm

    structured_llm = get_structured_llm(ComplianceReport)
    response = structured_llm.invoke(
        [
            {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Claim ID: {claim.claim_id}\n"
                    f"Clinical summary: {clinical_summary.model_dump_json() if clinical_summary else 'N/A'}\n"
                    f"Coding result: {coding_result.model_dump_json() if coding_result else 'N/A'}\n"
                    f"Validation result: {validation_result.model_dump_json() if validation_result else 'N/A'}\n"
                    f"Insurance findings: {insurance_findings.model_dump_json() if insurance_findings else 'N/A'}\n"
                    f"Revision cycles already run: {claim.revision_count}\n"
                    f"Prior agent activity log: {prior_logs}"
                ),
            },
        ]
    )
    # Always ground the audit trail in what actually happened, even in
    # live mode, rather than trusting the model to reproduce it verbatim.
    response.audit_trail = _build_audit_trail(
        claim, clinical_summary, coding_result, validation_result, insurance_findings, prior_logs
    )
    return response


def run_compliance_agent(
    claim: Claim,
    clinical_summary: ClinicalSummary | None,
    coding_result: CodingResult | None,
    validation_result: ValidationResult | None,
    insurance_findings: InsuranceFindings | None,
    prior_logs: list[str] | None = None,
) -> ComplianceReport:
    start = time.time()
    logger.info("[%s] Reviewing claim %s", AGENT_NAME, claim.claim_id)
    prior_logs = prior_logs or []

    if settings.use_mock_llm:
        result = _mock_review(claim, clinical_summary, coding_result, validation_result, insurance_findings, prior_logs)
    else:
        try:
            result = _llm_review(claim, clinical_summary, coding_result, validation_result, insurance_findings, prior_logs)
        except Exception:
            logger.exception("[%s] LLM review failed, falling back to mock", AGENT_NAME)
            result = _mock_review(claim, clinical_summary, coding_result, validation_result, insurance_findings, prior_logs)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim.claim_id,
            agent_name=AGENT_NAME,
            action="compliance_review",
            input_summary=f"revision_count={claim.revision_count}",
            output_summary=f"is_compliant={result.is_compliant}, human_review_required={result.human_review_required}, "
            f"findings={len(result.findings)}",
            duration_ms=duration_ms,
        )
    )
    return result
