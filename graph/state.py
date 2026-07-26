"""
Shared LangGraph state for the claim intake workflow and the denial /
appeal workflow.

LangGraph passes a single state object between nodes; each agent node
reads what it needs and returns a partial update that LangGraph merges
into the running state. Using a TypedDict (rather than a bare dict)
keeps the graph self-documenting and type-checkable.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from models.schemas import (
    AppealPackage,
    Claim,
    ClinicalSummary,
    CodingResult,
    ComplianceReport,
    DenialAnalysis,
    InsuranceFindings,
    ManagerAssessment,
    ValidationResult,
)


def _append(left: list, right: list) -> list:
    return left + right


class ClaimWorkflowState(TypedDict, total=False):
    """State for the primary claim-intake workflow. This is no longer a
    straight pipeline: Coding <-> Validation forms a feedback loop, the
    Compliance Agent acts as a quality gate before the Revenue Cycle
    Manager makes the final call, and the Clinical Documentation step can
    pause the whole graph on a LangGraph interrupt() if documentation is
    too sparse to proceed.

    Received -> Clinical Doc [-> pause for docs -> resume] -> Coding
      <-> Validation (feedback loop, bounded by max_coding_revisions)
      -> Insurance Policy (RAG) -> Compliance (quality gate)
      -> Revenue Cycle Manager (orchestrator) -> [Submit | Human Review]
    """

    claim: Claim
    clinical_summary: ClinicalSummary | None
    coding_result: CodingResult | None
    validation_result: ValidationResult | None
    insurance_findings: InsuranceFindings | None
    compliance_report: ComplianceReport | None
    manager_assessment: ManagerAssessment | None

    # Coding <-> Validation feedback loop control
    revision_count: int
    coding_feedback: str | None

    # Orchestration / shared-state metadata every agent reads and updates
    current_event: str
    workflow_stage: str
    agent_decisions: Annotated[list[str], _append]

    # Routing signal set by the Revenue Cycle Manager
    route_decision: str  # "ready_for_submission" | "needs_human_review"

    # Accumulated human-readable trace of what each agent did, rendered
    # in the Streamlit "Agent Activity" page and consumed by the
    # Compliance Agent to build its audit trail.
    logs: Annotated[list[str], _append]


class DenialWorkflowState(TypedDict, total=False):
    """State for the secondary denial/appeal workflow:
    Insurance Denial -> Denial Analysis -> [appealable?] -> Insurance
    Policy -> Compliance (quality gate) -> Appeal Generation ->
    Revenue Cycle Manager (orchestrator) -> Ready For Resubmission
    """

    claim: Claim
    denial_analysis: DenialAnalysis | None
    insurance_findings: InsuranceFindings | None
    compliance_report: ComplianceReport | None
    appeal_package: AppealPackage | None
    manager_assessment: ManagerAssessment | None

    current_event: str
    workflow_stage: str
    agent_decisions: Annotated[list[str], _append]

    route_decision: str  # "ready_for_resubmission" | "needs_human_review"

    logs: Annotated[list[str], _append]
