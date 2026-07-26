"""
Repository layer.

Encapsulates all SQL so agents, tools, and the Streamlit UI never write
raw queries. Pydantic models are serialized to/from JSON columns for the
nested agent-output fields.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from database.db import db_session
from models.schemas import (
    AgentLogEntry,
    AppealPackage,
    Claim,
    ClaimEvent,
    ClaimStatus,
    ClinicalSummary,
    CodingResult,
    ComplianceReport,
    DenialAnalysis,
    InsuranceFindings,
    ManagerAssessment,
    Patient,
    RiskLevel,
    ValidationResult,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_NESTED_FIELDS = {
    "clinical_summary_json": ("clinical_summary", ClinicalSummary),
    "coding_result_json": ("coding_result", CodingResult),
    "validation_result_json": ("validation_result", ValidationResult),
    "insurance_findings_json": ("insurance_findings", InsuranceFindings),
    "compliance_report_json": ("compliance_report", ComplianceReport),
    "denial_analysis_json": ("denial_analysis", DenialAnalysis),
    "appeal_package_json": ("appeal_package", AppealPackage),
    "manager_assessment_json": ("manager_assessment", ManagerAssessment),
}


def _row_to_claim(row: sqlite3.Row) -> Claim:
    data = dict(row)
    nested_kwargs = {}
    for json_col, (field_name, model_cls) in _NESTED_FIELDS.items():
        raw = data.pop(json_col, None)
        nested_kwargs[field_name] = model_cls.model_validate_json(raw) if raw else None

    audit_trail_raw = data.pop("audit_trail_json", None)
    nested_kwargs["audit_trail"] = json.loads(audit_trail_raw) if audit_trail_raw else []

    return Claim(**data, **nested_kwargs)


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

def save_patient(patient: Patient) -> None:
    with db_session() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO patients
               (patient_id, first_name, last_name, date_of_birth, payer, member_id, plan_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                patient.patient_id,
                patient.first_name,
                patient.last_name,
                patient.date_of_birth,
                patient.payer.value,
                patient.member_id,
                patient.plan_name,
            ),
        )


def get_patient(patient_id: str) -> Patient | None:
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
    return Patient(**dict(row)) if row else None


def list_patients() -> list[Patient]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM patients ORDER BY last_name").fetchall()
    return [Patient(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

def save_claim(claim: Claim) -> None:
    claim.updated_at = datetime.utcnow()
    with db_session() as conn:
        conn.execute(
            """INSERT INTO claims (
                claim_id, patient_id, patient_name, payer, date_of_service,
                billed_amount, physician_notes_raw, status, risk_level, denial_reason,
                clinical_summary_json, coding_result_json, validation_result_json,
                insurance_findings_json, compliance_report_json, denial_analysis_json,
                appeal_package_json, manager_assessment_json, workflow_stage,
                revision_count, audit_trail_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                patient_id=excluded.patient_id,
                patient_name=excluded.patient_name,
                payer=excluded.payer,
                date_of_service=excluded.date_of_service,
                billed_amount=excluded.billed_amount,
                physician_notes_raw=excluded.physician_notes_raw,
                status=excluded.status,
                risk_level=excluded.risk_level,
                denial_reason=excluded.denial_reason,
                clinical_summary_json=excluded.clinical_summary_json,
                coding_result_json=excluded.coding_result_json,
                validation_result_json=excluded.validation_result_json,
                insurance_findings_json=excluded.insurance_findings_json,
                compliance_report_json=excluded.compliance_report_json,
                denial_analysis_json=excluded.denial_analysis_json,
                appeal_package_json=excluded.appeal_package_json,
                manager_assessment_json=excluded.manager_assessment_json,
                workflow_stage=excluded.workflow_stage,
                revision_count=excluded.revision_count,
                audit_trail_json=excluded.audit_trail_json,
                updated_at=excluded.updated_at
            """,
            (
                claim.claim_id,
                claim.patient_id,
                claim.patient_name,
                claim.payer.value,
                claim.date_of_service,
                claim.billed_amount,
                claim.physician_notes_raw,
                claim.status.value,
                claim.risk_level.value,
                claim.denial_reason,
                claim.clinical_summary.model_dump_json() if claim.clinical_summary else None,
                claim.coding_result.model_dump_json() if claim.coding_result else None,
                claim.validation_result.model_dump_json() if claim.validation_result else None,
                claim.insurance_findings.model_dump_json() if claim.insurance_findings else None,
                claim.compliance_report.model_dump_json() if claim.compliance_report else None,
                claim.denial_analysis.model_dump_json() if claim.denial_analysis else None,
                claim.appeal_package.model_dump_json() if claim.appeal_package else None,
                claim.manager_assessment.model_dump_json() if claim.manager_assessment else None,
                claim.workflow_stage,
                claim.revision_count,
                json.dumps(claim.audit_trail),
                claim.created_at.isoformat(),
                claim.updated_at.isoformat(),
            ),
        )
    logger.info("Saved claim %s (status=%s, stage=%s)", claim.claim_id, claim.status.value, claim.workflow_stage)


def get_claim(claim_id: str) -> Claim | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
    return _row_to_claim(row) if row else None


def list_claims(status: ClaimStatus | None = None, risk_level: RiskLevel | None = None) -> list[Claim]:
    query = "SELECT * FROM claims"
    conditions, params = [], []
    if status:
        conditions.append("status = ?")
        params.append(status.value)
    if risk_level:
        conditions.append("risk_level = ?")
        params.append(risk_level.value)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY updated_at DESC"

    with db_session() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_claim(r) for r in rows]


def find_duplicate_claim(patient_id: str, date_of_service: str, exclude_claim_id: str | None = None) -> Claim | None:
    """Look for an existing claim for the same patient/date-of-service."""
    with db_session() as conn:
        row = conn.execute(
            """SELECT * FROM claims
               WHERE patient_id = ? AND date_of_service = ? AND claim_id != ?
               LIMIT 1""",
            (patient_id, date_of_service, exclude_claim_id or ""),
        ).fetchone()
    return _row_to_claim(row) if row else None


# ---------------------------------------------------------------------------
# Appeals
# ---------------------------------------------------------------------------

def save_appeal(claim_id: str, appeal: AppealPackage) -> None:
    with db_session() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO appeals
               (appeal_id, claim_id, appeal_letter, missing_documentation_checklist,
                supporting_evidence_summary, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                appeal.appeal_id,
                claim_id,
                appeal.appeal_letter,
                "\n".join(appeal.missing_documentation_checklist),
                appeal.supporting_evidence_summary,
                appeal.status,
                appeal.created_at.isoformat(),
            ),
        )


def list_appeals() -> list[dict]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM appeals ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Agent logs
# ---------------------------------------------------------------------------

def log_agent_action(entry: AgentLogEntry) -> None:
    with db_session() as conn:
        conn.execute(
            """INSERT INTO agent_logs
               (log_id, claim_id, agent_name, action, input_summary, output_summary,
                duration_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.log_id,
                entry.claim_id,
                entry.agent_name,
                entry.action,
                entry.input_summary,
                entry.output_summary,
                entry.duration_ms,
                entry.created_at.isoformat(),
            ),
        )


def list_agent_logs(claim_id: str | None = None, limit: int = 200) -> list[dict]:
    query = "SELECT * FROM agent_logs"
    params: list = []
    if claim_id:
        query += " WHERE claim_id = ?"
        params.append(claim_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db_session() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def save_event(event: ClaimEvent) -> None:
    with db_session() as conn:
        conn.execute(
            """INSERT INTO events (event_id, claim_id, event_type, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.claim_id,
                event.event_type.value,
                json.dumps(event.payload),
                event.created_at.isoformat(),
            ),
        )
    logger.info("Event recorded: %s for claim %s", event.event_type.value, event.claim_id)


def list_events(claim_id: str | None = None, limit: int = 200) -> list[dict]:
    query = "SELECT * FROM events"
    params: list = []
    if claim_id:
        query += " WHERE claim_id = ?"
        params.append(claim_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db_session() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Dashboard aggregate stats
# ---------------------------------------------------------------------------

def get_dashboard_stats() -> dict:
    with db_session() as conn:
        claims_today = conn.execute(
            "SELECT COUNT(*) c FROM claims WHERE date(created_at) = date('now')"
        ).fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) c FROM claims WHERE status IN (?, ?)",
            (ClaimStatus.RECEIVED.value, ClaimStatus.IN_REVIEW.value),
        ).fetchone()["c"]
        denied = conn.execute(
            "SELECT COUNT(*) c FROM claims WHERE status = ?", (ClaimStatus.DENIED.value,)
        ).fetchone()["c"]
        appeals_pending = conn.execute(
            "SELECT COUNT(*) c FROM appeals WHERE status = 'drafted'"
        ).fetchone()["c"]
        submission_ready = conn.execute(
            "SELECT COUNT(*) c FROM claims WHERE status = ?",
            (ClaimStatus.READY_FOR_SUBMISSION.value,),
        ).fetchone()["c"]
        high_risk = conn.execute(
            "SELECT COUNT(*) c FROM claims WHERE risk_level = ?", (RiskLevel.HIGH.value,)
        ).fetchone()["c"]
        avg_conf_row = conn.execute(
            "SELECT coding_result_json FROM claims WHERE coding_result_json IS NOT NULL"
        ).fetchall()

    import json as _json

    confidences = []
    for r in avg_conf_row:
        try:
            confidences.append(_json.loads(r["coding_result_json"])["overall_confidence"])
        except Exception:
            continue
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return {
        "claims_today": claims_today,
        "pending_claims": pending,
        "denied_claims": denied,
        "appeals_pending": appeals_pending,
        "submission_ready": submission_ready,
        "high_risk_claims": high_risk,
        "average_confidence": avg_confidence,
    }
