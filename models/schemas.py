"""
Pydantic data contracts shared across agents, tools, the database layer,
and the Streamlit UI.

Keeping these as the single source of truth prevents structural drift
between what an agent returns and what the graph state / database expects.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    RECEIVED = "received"
    AWAITING_DOCUMENTATION = "awaiting_documentation"
    IN_REVIEW = "in_review"
    READY_FOR_SUBMISSION = "ready_for_submission"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    SUBMITTED = "submitted"
    DENIED = "denied"
    APPEALING = "appealing"
    READY_FOR_RESUBMISSION = "ready_for_resubmission"
    RESUBMITTED = "resubmitted"
    APPROVED = "approved"
    CLOSED = "closed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventType(str, Enum):
    NEW_CLAIM = "new_claim"
    CLAIM_UPDATED = "claim_updated"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENTATION_REQUIRED = "documentation_required"
    CODING_COMPLETED = "coding_completed"
    CODING_REVISION_REQUESTED = "coding_revision_requested"
    VALIDATION_FAILED = "validation_failed"
    COMPLIANCE_FAILED = "compliance_failed"
    CLAIM_DENIED = "claim_denied"
    DOCUMENTATION_UPLOADED = "documentation_uploaded"
    APPEAL_REQUIRED = "appeal_required"
    APPEAL_SUBMITTED = "appeal_submitted"
    CLAIM_APPROVED = "claim_approved"
    CLAIM_READY_FOR_SUBMISSION = "claim_ready_for_submission"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    PAYMENT_RECEIVED = "payment_received"


class Payer(str, Enum):
    UNITEDHEALTHCARE = "UnitedHealthcare"
    AETNA = "Aetna"
    CIGNA = "Cigna"
    BCBS = "Blue Cross Blue Shield"


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class Patient(BaseModel):
    patient_id: str = Field(default_factory=lambda: _new_id("PT"))
    first_name: str
    last_name: str
    date_of_birth: str
    payer: Payer
    member_id: str
    plan_name: str = "Standard PPO"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


# ---------------------------------------------------------------------------
# Clinical Documentation Agent output
# ---------------------------------------------------------------------------

class ClinicalSummary(BaseModel):
    diagnoses: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    chief_complaint: str | None = None
    physician_name: str | None = None
    date_of_service: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    summary_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# ---------------------------------------------------------------------------
# Medical Coding Agent output
# ---------------------------------------------------------------------------

class CodeRecommendation(BaseModel):
    code: str
    description: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class CodingResult(BaseModel):
    icd10_codes: list[CodeRecommendation] = Field(default_factory=list)
    cpt_codes: list[CodeRecommendation] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    coding_notes: str = ""


# ---------------------------------------------------------------------------
# Claim Validation Agent output
# ---------------------------------------------------------------------------

class ValidationIssue(BaseModel):
    field: str
    issue: str
    severity: str = Field(description="one of: info, warning, critical")


class ValidationResult(BaseModel):
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    duplicate_claim_detected: bool = False
    completeness_score: float = Field(ge=0.0, le=1.0, default=0.0)


# ---------------------------------------------------------------------------
# Insurance Policy Agent (RAG) output
# ---------------------------------------------------------------------------

class PolicyReference(BaseModel):
    payer: str
    document: str
    excerpt: str


class InsuranceFindings(BaseModel):
    is_covered: bool
    prior_authorization_required: bool
    required_documents: list[str] = Field(default_factory=list)
    policy_references: list[PolicyReference] = Field(default_factory=list)
    rationale: str = Field(
        default="", description="Explanation of why the cited policy applies to this claim"
    )
    conflicting_rules: list[str] = Field(
        default_factory=list,
        description="Any contradictions found across retrieved policy excerpts, if present",
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Denial Analysis Agent output
# ---------------------------------------------------------------------------

class DenialAnalysis(BaseModel):
    denial_reason_code: str
    root_cause: str
    category: str = Field(
        description="one of: coding_error, missing_documentation, "
        "prior_auth, coverage_exclusion, duplicate, eligibility, other"
    )
    recommended_corrections: list[str] = Field(default_factory=list)
    is_appealable: bool = True


# ---------------------------------------------------------------------------
# Compliance Agent output — the quality gate that reviews every other
# agent's output before a claim can move to submission or resubmission.
# ---------------------------------------------------------------------------

class ComplianceFinding(BaseModel):
    area: str = Field(description="which upstream agent/area this finding relates to, "
                       "e.g. 'coding', 'documentation', 'prior_authorization'")
    finding: str
    severity: str = Field(description="one of: info, warning, critical")


class ComplianceReport(BaseModel):
    is_compliant: bool
    findings: list[ComplianceFinding] = Field(default_factory=list)
    human_review_required: bool = False
    audit_trail: list[str] = Field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Appeal Generation Agent output
# ---------------------------------------------------------------------------

class AppealPackage(BaseModel):
    appeal_id: str = Field(default_factory=lambda: _new_id("APL"))
    appeal_letter: str
    missing_documentation_checklist: list[str] = Field(default_factory=list)
    supporting_evidence_summary: str
    status: str = "drafted"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Revenue Cycle Manager Agent output
# ---------------------------------------------------------------------------

class RecommendedAction(BaseModel):
    claim_id: str
    action: str
    priority: str = Field(description="one of: low, medium, high, urgent")
    rationale: str


class ManagerAssessment(BaseModel):
    risk_level: RiskLevel
    priority_score: float = Field(ge=0.0, le=100.0)
    recommended_action: str
    rationale: str
    next_step: str = Field(
        default="", description="the concrete workflow directive the manager is issuing, "
        "e.g. 'submit_claim', 'escalate_to_human_review', 'request_documentation'"
    )
    escalated: bool = False


# ---------------------------------------------------------------------------
# Claim (top-level aggregate)
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: _new_id("CLM"))
    patient_id: str
    patient_name: str
    payer: Payer
    date_of_service: str
    billed_amount: float = Field(ge=0)
    physician_notes_raw: str
    status: ClaimStatus = ClaimStatus.RECEIVED
    risk_level: RiskLevel = RiskLevel.LOW
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    clinical_summary: ClinicalSummary | None = None
    coding_result: CodingResult | None = None
    validation_result: ValidationResult | None = None
    insurance_findings: InsuranceFindings | None = None
    compliance_report: ComplianceReport | None = None
    denial_analysis: DenialAnalysis | None = None
    appeal_package: AppealPackage | None = None
    manager_assessment: ManagerAssessment | None = None

    denial_reason: str | None = None

    # Multi-agent collaboration metadata
    workflow_stage: str = "intake"
    revision_count: int = 0
    audit_trail: list[str] = Field(default_factory=list)

    @field_validator("billed_amount")
    @classmethod
    def _round_amount(cls, v: float) -> float:
        return round(v, 2)


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

class ClaimEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: _new_id("EVT"))
    claim_id: str
    event_type: EventType
    payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Agent log entry
# ---------------------------------------------------------------------------

class AgentLogEntry(BaseModel):
    log_id: str = Field(default_factory=lambda: _new_id("LOG"))
    claim_id: str
    agent_name: str
    action: str
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
