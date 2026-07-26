"""Appeal generation tool — produces an insurance appeal letter draft."""

from __future__ import annotations

from datetime import date

from langchain_core.tools import tool

from database import repository
from models.schemas import AppealPackage, Claim, DenialAnalysis, InsuranceFindings


def generate_appeal_letter(
    claim: Claim, denial_analysis: DenialAnalysis, insurance_findings: InsuranceFindings | None
) -> str:
    """Deterministic template-based appeal letter generator (mock-mode
    fallback; also used as the base structure for the LLM-enhanced path)."""
    today = date.today().strftime("%B %d, %Y")
    policy_refs = ""
    if insurance_findings and insurance_findings.policy_references:
        policy_refs = "\n".join(
            f"  - {p.document} ({p.payer}): {p.excerpt}"
            for p in insurance_findings.policy_references
        )

    corrections = "\n".join(f"  - {c}" for c in denial_analysis.recommended_corrections)

    return f"""{today}

RE: Formal Appeal of Claim Denial
Claim ID: {claim.claim_id}
Patient: {claim.patient_name}
Date of Service: {claim.date_of_service}
Payer: {claim.payer.value}

To the {claim.payer.value} Appeals Department,

This letter serves as a formal appeal of the denial issued for the
above-referenced claim, denied under reason code {denial_analysis.denial_reason_code}.

Root Cause of Denial:
{denial_analysis.root_cause}

Basis for Appeal:
The denial was issued due to a {denial_analysis.category.replace('_', ' ')} classification.
We have addressed this as follows:
{corrections or '  - See attached corrected documentation.'}

{"Supporting Policy References:" if policy_refs else ""}
{policy_refs}

We respectfully request that {claim.payer.value} reverse the denial and
reprocess this claim for payment based on the corrected information and
supporting documentation enclosed with this appeal.

Please contact our billing department with any questions regarding this
appeal or if additional documentation is required.

Sincerely,
Revenue Cycle Management Team
"""


def generate_missing_documentation_checklist(denial_analysis: DenialAnalysis) -> list[str]:
    base_checklist = list(denial_analysis.recommended_corrections)
    if denial_analysis.category == "missing_documentation":
        base_checklist.append("Signed physician progress notes for date of service")
        base_checklist.append("Operative or procedure report (if applicable)")
    if denial_analysis.category == "prior_auth":
        base_checklist.append("Prior authorization request confirmation or reference number")
        base_checklist.append("Letter of medical necessity")
    if denial_analysis.category == "coding_error":
        base_checklist.append("Corrected claim form with updated ICD-10/CPT codes")
    return list(dict.fromkeys(base_checklist))  # de-duplicate, preserve order


@tool("appeal_generator_tool")
def appeal_generator_tool(claim_id: str) -> str:
    """Generate a formal insurance appeal letter and supporting checklist
    for a denied claim that already has denial analysis on file."""
    claim = repository.get_claim(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."
    if not claim.denial_analysis:
        return f"Claim {claim_id} has no denial analysis on file. Run the Denial Analysis Agent first."

    letter = generate_appeal_letter(claim, claim.denial_analysis, claim.insurance_findings)
    checklist = generate_missing_documentation_checklist(claim.denial_analysis)
    package = AppealPackage(
        appeal_letter=letter,
        missing_documentation_checklist=checklist,
        supporting_evidence_summary=(
            f"Denial category: {claim.denial_analysis.category}. "
            f"Root cause: {claim.denial_analysis.root_cause}"
        ),
    )
    repository.save_appeal(claim.claim_id, package)
    return f"Appeal {package.appeal_id} generated for claim {claim_id}.\n\n{letter}"
