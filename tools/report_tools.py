"""Report generation tool — produces Markdown summaries for claims and the
overall revenue cycle dashboard."""

from __future__ import annotations

from langchain_core.tools import tool

from database import repository


def generate_claim_report(claim_id: str) -> str:
    claim = repository.get_claim(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."

    lines = [
        f"# Claim Report — {claim.claim_id}",
        "",
        f"**Patient:** {claim.patient_name}  ",
        f"**Payer:** {claim.payer.value}  ",
        f"**Date of Service:** {claim.date_of_service}  ",
        f"**Billed Amount:** ${claim.billed_amount:,.2f}  ",
        f"**Status:** {claim.status.value}  ",
        f"**Risk Level:** {claim.risk_level.value}  ",
        "",
    ]

    if claim.clinical_summary:
        lines += [
            "## Clinical Summary",
            f"- Diagnoses: {', '.join(claim.clinical_summary.diagnoses) or 'None extracted'}",
            f"- Procedures: {', '.join(claim.clinical_summary.procedures) or 'None extracted'}",
            f"- Missing information: {', '.join(claim.clinical_summary.missing_information) or 'None'}",
            "",
        ]

    if claim.coding_result:
        icd = ", ".join(f"{c.code} ({c.description})" for c in claim.coding_result.icd10_codes)
        cpt = ", ".join(f"{c.code} ({c.description})" for c in claim.coding_result.cpt_codes)
        lines += [
            "## Coding",
            f"- ICD-10: {icd or 'None assigned'}",
            f"- CPT: {cpt or 'None assigned'}",
            f"- Overall confidence: {claim.coding_result.overall_confidence:.2f}",
            "",
        ]

    if claim.validation_result:
        lines += [
            "## Validation",
            f"- Valid: {claim.validation_result.is_valid}",
            f"- Completeness score: {claim.validation_result.completeness_score:.2f}",
            f"- Duplicate detected: {claim.validation_result.duplicate_claim_detected}",
        ]
        for issue in claim.validation_result.issues:
            lines.append(f"  - [{issue.severity}] {issue.field}: {issue.issue}")
        lines.append("")

    if claim.insurance_findings:
        lines += [
            "## Insurance Findings",
            f"- Covered: {claim.insurance_findings.is_covered}",
            f"- Prior authorization required: {claim.insurance_findings.prior_authorization_required}",
            f"- Required documents: {', '.join(claim.insurance_findings.required_documents) or 'None'}",
            "",
        ]

    if claim.denial_analysis:
        lines += [
            "## Denial Analysis",
            f"- Reason code: {claim.denial_analysis.denial_reason_code}",
            f"- Category: {claim.denial_analysis.category}",
            f"- Root cause: {claim.denial_analysis.root_cause}",
            f"- Appealable: {claim.denial_analysis.is_appealable}",
            "",
        ]

    if claim.manager_assessment:
        lines += [
            "## Revenue Cycle Manager Assessment",
            f"- Risk level: {claim.manager_assessment.risk_level.value}",
            f"- Priority score: {claim.manager_assessment.priority_score:.0f}/100",
            f"- Recommended action: {claim.manager_assessment.recommended_action}",
            f"- Rationale: {claim.manager_assessment.rationale}",
        ]

    return "\n".join(lines)


def generate_dashboard_report() -> str:
    stats = repository.get_dashboard_stats()
    lines = [
        "# Revenue Cycle Dashboard Report",
        "",
        f"- Claims today: {stats['claims_today']}",
        f"- Pending claims: {stats['pending_claims']}",
        f"- Denied claims: {stats['denied_claims']}",
        f"- Appeals pending: {stats['appeals_pending']}",
        f"- Submission ready: {stats['submission_ready']}",
        f"- High risk claims: {stats['high_risk_claims']}",
        f"- Average coding confidence: {stats['average_confidence']}",
    ]
    return "\n".join(lines)


@tool("report_generator_tool")
def report_generator_tool(claim_id: str = "") -> str:
    """Generate a Markdown report for a specific claim, or the overall
    revenue cycle dashboard when no claim_id is provided."""
    if claim_id:
        return generate_claim_report(claim_id)
    return generate_dashboard_report()
