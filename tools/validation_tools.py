"""
Claim validation tool.

Implements deterministic rule checks for claim completeness, duplicate
detection, and coding consistency. Used directly by the Claim
Validation Agent (rule-based core) and exposed as a LangChain tool so
an LLM-driven agent variant can also invoke it.
"""

from __future__ import annotations

from langchain_core.tools import tool

from database import repository
from models.schemas import Claim, CodingResult, ValidationIssue, ValidationResult

REQUIRED_FIELDS = ["patient_id", "payer", "date_of_service", "billed_amount"]


def run_claim_validation(claim: Claim, coding_result: CodingResult | None) -> ValidationResult:
    """Core rule-based validation logic, callable directly from Python."""
    issues: list[ValidationIssue] = []

    for field_name in REQUIRED_FIELDS:
        value = getattr(claim, field_name, None)
        if value in (None, "", 0):
            issues.append(
                ValidationIssue(field=field_name, issue="Required field is missing", severity="critical")
            )

    if claim.billed_amount <= 0:
        issues.append(
            ValidationIssue(field="billed_amount", issue="Billed amount must be greater than zero", severity="critical")
        )

    duplicate = repository.find_duplicate_claim(
        claim.patient_id, claim.date_of_service, exclude_claim_id=claim.claim_id
    )
    duplicate_detected = duplicate is not None
    if duplicate_detected:
        issues.append(
            ValidationIssue(
                field="claim",
                issue=f"Possible duplicate of existing claim {duplicate.claim_id} "
                f"(same patient and date of service)",
                severity="warning",
            )
        )

    if coding_result is None or not coding_result.icd10_codes:
        issues.append(
            ValidationIssue(field="icd10_codes", issue="No ICD-10 diagnosis codes assigned", severity="critical")
        )
    if coding_result is None or not coding_result.cpt_codes:
        issues.append(
            ValidationIssue(field="cpt_codes", issue="No CPT procedure codes assigned", severity="critical")
        )
    if coding_result and coding_result.overall_confidence < 0.6:
        issues.append(
            ValidationIssue(
                field="coding_result",
                issue=f"Coding confidence is low ({coding_result.overall_confidence:.2f})",
                severity="warning",
            )
        )

    critical_count = sum(1 for i in issues if i.severity == "critical")
    total_checks = len(REQUIRED_FIELDS) + 3
    completeness_score = max(0.0, 1.0 - (critical_count / total_checks))

    return ValidationResult(
        is_valid=critical_count == 0 and not duplicate_detected,
        issues=issues,
        duplicate_claim_detected=duplicate_detected,
        completeness_score=round(completeness_score, 2),
    )


@tool("claim_validation_tool")
def claim_validation_tool(claim_id: str) -> str:
    """Run rule-based completeness, duplicate, and coding-consistency
    validation against a stored claim and return a human-readable summary."""
    claim = repository.get_claim(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."
    result = run_claim_validation(claim, claim.coding_result)
    issue_lines = "\n".join(f"- [{i.severity}] {i.field}: {i.issue}" for i in result.issues) or "None"
    return (
        f"Valid: {result.is_valid} | Completeness: {result.completeness_score} | "
        f"Duplicate: {result.duplicate_claim_detected}\nIssues:\n{issue_lines}"
    )
