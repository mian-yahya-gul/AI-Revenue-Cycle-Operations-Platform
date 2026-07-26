"""
LangGraph orchestration.

This is not a linear pipeline. Agents collaborate through shared state
and conditional routing rather than calling each other directly:

- **Coding <-> Validation feedback loop**: if Claim Validation finds the
  Medical Coding Agent's output inadequate, it sends the claim back for
  a bounded number of revisions (settings.max_coding_revisions) before
  giving up and letting the Compliance Agent flag it for a human.
- **Compliance quality gate**: after Insurance Policy runs, the
  Compliance Agent reviews every prior agent's output as a whole and
  produces the single authoritative human-review determination.
- **Revenue Cycle Manager as orchestrator**: the manager treats the
  Compliance Agent's verdict as a hard business rule and issues the
  final routing decision, rather than the graph deciding independently.
- **Documentation-pause interrupt**: if the Clinical Documentation Agent
  cannot extract anything usable at all, the graph pauses via LangGraph's
  interrupt() — checkpointed so it can be resumed later, from exactly
  where it left off, once a human supplies additional documentation.

Two graphs are defined:

1. ``build_claim_intake_graph`` — Claim Received -> Clinical Documentation
   [pause/resume on sparse notes] -> Coding <-> Validation (loop) ->
   Insurance Policy (RAG) -> Compliance -> Revenue Cycle Manager (final
   routing) -> Ready for Submission | Needs Human Review.

2. ``build_denial_appeal_graph`` — Insurance Denial -> Denial Analysis
   -> [appealable?] -> Insurance Policy -> Compliance -> Appeal
   Generation -> Revenue Cycle Manager -> Ready for Resubmission, or
   Needs Human Review at either gate.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents.appeal_generation_agent import run_appeal_generation_agent
from agents.claim_validation_agent import run_claim_validation_agent
from agents.clinical_documentation_agent import run_clinical_documentation_agent
from agents.compliance_agent import run_compliance_agent
from agents.denial_analysis_agent import run_denial_analysis_agent
from agents.insurance_policy_agent import run_insurance_policy_agent
from agents.medical_coding_agent import run_medical_coding_agent
from agents.revenue_cycle_manager_agent import run_revenue_cycle_manager_agent
from database.repository import save_appeal, save_claim, save_event
from graph.routing import (
    route_after_compliance_denial,
    route_after_denial_analysis,
    route_after_validation,
    determine_final_route,
)
from graph.state import ClaimWorkflowState, DenialWorkflowState
from models.schemas import ClaimEvent, ClaimStatus, EventType
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Claim intake workflow nodes
# ---------------------------------------------------------------------------

def _documentation_insufficient(summary) -> bool:
    """The bar for pausing the whole workflow: nothing at all was
    extractable from the notes. Partial documentation still proceeds —
    that's what the Claim Validation / Compliance stages are for — but a
    genuinely empty extraction means there's nothing for any downstream
    agent to work with."""
    return not summary.diagnoses and not summary.procedures


def _node_clinical_documentation(state: ClaimWorkflowState) -> dict:
    from config.settings import settings
    from database.repository import get_claim as _get_persisted_claim

    claim = state["claim"]
    summary = run_clinical_documentation_agent(claim.claim_id, claim.physician_notes_raw)

    # Count how many document-update cycles this claim has already
    # committed, once, before entering the loop below. Because LangGraph
    # replays every earlier-resolved interrupt in this node from the top
    # on each new resume, comparing against "current" DB content is
    # unreliable once there's more than one loop iteration (the DB has
    # already moved past what an earlier replay step represents). A
    # stable count of what's already been committed, matched against
    # this replay's iteration position, is the correct idempotency key:
    # only the iteration whose index equals the existing count is the
    # genuinely new one this invoke owns.
    from database.repository import list_events as _list_events

    already_committed_updates = sum(
        1 for e in _list_events(claim_id=claim.claim_id) if e["event_type"] == EventType.DOCUMENT_UPDATED.value
    )
    resume_cycle_index = 0

    while settings.documentation_pause_enabled and _documentation_insufficient(summary):
        # LangGraph re-executes a node's code from the top on every
        # resume attempt, replaying earlier (already-resolved) interrupt()
        # calls before it reaches a new, unresolved one. That means any
        # side effect in this loop can run more than once across nested
        # resumes — so every save/event here is gated on a comparison
        # against durable DB state (not local variables, which get
        # reconstructed fresh on every replay) to stay idempotent.
        persisted = _get_persisted_claim(claim.claim_id)
        already_paused = persisted is not None and persisted.status == ClaimStatus.AWAITING_DOCUMENTATION

        # Keep the in-memory object's status consistent even on a replay,
        # where `claim` is a fresh copy deserialized from the checkpoint
        # (still carrying its pre-pause status) rather than the mutated
        # object from a previous invoke — otherwise a later save_claim()
        # call in this same iteration could write that stale status back
        # over the DB and silently un-pause the claim.
        claim.clinical_summary = summary
        claim.status = ClaimStatus.AWAITING_DOCUMENTATION
        claim.workflow_stage = "awaiting_documentation"

        if not already_paused:
            save_claim(claim)
            save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.DOCUMENTATION_REQUIRED))
            logger.info("Claim %s paused — insufficient documentation to proceed", claim.claim_id)

        # Pause the graph here. Execution (and this node) resumes from
        # this exact point when someone calls Command(resume=...) against
        # this thread_id — `resume_payload` below is that resume value.
        resume_payload = interrupt(
            {
                "reason": "insufficient_documentation",
                "claim_id": claim.claim_id,
                "missing_information": summary.missing_information,
                "message": "Clinical Documentation Agent could not extract any diagnoses or "
                "procedures from the physician notes provided. Upload additional "
                "documentation to continue.",
            }
        )

        updated_notes = resume_payload.get("updated_notes") if isinstance(resume_payload, dict) else None

        if updated_notes:
            claim.physician_notes_raw = updated_notes
            if resume_cycle_index >= already_committed_updates:
                # This is the one genuinely new resolution in this
                # invoke — every other iteration in this replay is
                # re-walking history that was already committed by an
                # earlier invoke.
                save_claim(claim)
                save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.DOCUMENT_UPDATED))
        resume_cycle_index += 1

        # Only the affected agent (Clinical Documentation) re-runs here;
        # everything downstream re-runs naturally as the graph continues
        # because it all depends on this corrected summary. If the
        # updated notes are STILL insufficient, the while condition
        # re-evaluates and the graph pauses again rather than forcing a
        # claim with nothing usable in it through the rest of the team.
        summary = run_clinical_documentation_agent(claim.claim_id, claim.physician_notes_raw)

    claim.clinical_summary = summary
    claim.status = ClaimStatus.IN_REVIEW
    claim.workflow_stage = "clinical_documentation"
    save_claim(claim)
    return {
        "claim": claim,
        "clinical_summary": summary,
        "workflow_stage": "clinical_documentation",
        "current_event": EventType.CLAIM_UPDATED.value,
        "logs": [f"Clinical Documentation Agent extracted {len(summary.diagnoses)} diagnoses, "
                 f"{len(summary.procedures)} procedures (confidence={summary.confidence:.2f})"],
    }


def _node_medical_coding(state: ClaimWorkflowState) -> dict:
    claim = state["claim"]
    feedback = state.get("coding_feedback")
    revision_count = state.get("revision_count", 0)

    coding_result = run_medical_coding_agent(claim.claim_id, state["clinical_summary"], feedback=feedback)
    claim.coding_result = coding_result
    claim.workflow_stage = "medical_coding_revision" if feedback else "medical_coding"
    if feedback:
        claim.revision_count = revision_count + 1
        save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.CODING_REVISION_REQUESTED))
    save_claim(claim)

    log_line = (
        f"Medical Coding Agent revised coding (attempt {claim.revision_count}) in response to "
        f"validation feedback: {feedback}"
        if feedback else
        f"Medical Coding Agent assigned {len(coding_result.icd10_codes)} ICD-10 and "
        f"{len(coding_result.cpt_codes)} CPT codes (confidence={coding_result.overall_confidence:.2f})"
    )

    return {
        "claim": claim,
        "coding_result": coding_result,
        "revision_count": claim.revision_count,
        "coding_feedback": None,
        "workflow_stage": claim.workflow_stage,
        "current_event": EventType.CODING_COMPLETED.value,
        "logs": [log_line],
    }


def _node_claim_validation(state: ClaimWorkflowState) -> dict:
    from graph.routing import _coding_needs_revision

    claim = state["claim"]
    validation_result = run_claim_validation_agent(claim, state["coding_result"])
    claim.validation_result = validation_result
    claim.workflow_stage = "claim_validation"
    save_claim(claim)

    needs_revision, feedback = _coding_needs_revision({**state, "coding_result": claim.coding_result})
    if not validation_result.is_valid:
        save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.VALIDATION_FAILED))

    result = {
        "claim": claim,
        "validation_result": validation_result,
        "workflow_stage": "claim_validation",
        "current_event": EventType.VALIDATION_FAILED.value if not validation_result.is_valid else "validation_passed",
        "logs": [f"Claim Validation Agent found {len(validation_result.issues)} issue(s); "
                 f"valid={validation_result.is_valid}"],
    }
    # Stash feedback for the (possible) revision hop back to Medical Coding.
    # route_after_validation independently decides whether the loop
    # actually fires (respecting the revision budget) — this is just the
    # message Medical Coding will see if it does.
    if needs_revision:
        result["coding_feedback"] = feedback
        result["agent_decisions"] = [
            f"Claim Validation Agent: requesting coding revision — {feedback}"
        ]
    else:
        result["agent_decisions"] = ["Claim Validation Agent: coding accepted, no revision needed"]
    return result


def _node_insurance_policy(state: ClaimWorkflowState) -> dict:
    claim = state["claim"]
    findings = run_insurance_policy_agent(claim, state["clinical_summary"])
    claim.insurance_findings = findings
    claim.workflow_stage = "insurance_policy"
    save_claim(claim)
    return {
        "claim": claim,
        "insurance_findings": findings,
        "workflow_stage": "insurance_policy",
        "current_event": EventType.CLAIM_UPDATED.value,
        "logs": [f"Insurance Policy Agent (RAG): covered={findings.is_covered}, "
                 f"prior_auth_required={findings.prior_authorization_required}, "
                 f"conflicts={len(findings.conflicting_rules)}"],
    }


def _node_compliance(state: ClaimWorkflowState) -> dict:
    claim = state["claim"]
    report = run_compliance_agent(
        claim,
        state.get("clinical_summary"),
        state.get("coding_result"),
        state.get("validation_result"),
        state.get("insurance_findings"),
        prior_logs=state.get("logs", []),
    )
    claim.compliance_report = report
    claim.workflow_stage = "compliance_review"
    claim.audit_trail = report.audit_trail
    save_claim(claim)

    if not report.is_compliant:
        save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.COMPLIANCE_FAILED))

    return {
        "claim": claim,
        "compliance_report": report,
        "workflow_stage": "compliance_review",
        "current_event": EventType.COMPLIANCE_FAILED.value if not report.is_compliant else "compliance_passed",
        "logs": [f"Compliance Agent: is_compliant={report.is_compliant}, "
                 f"human_review_required={report.human_review_required}, findings={len(report.findings)}"],
        "agent_decisions": [
            f"Compliance Agent: {'BLOCKED — human review required' if report.human_review_required else 'cleared for submission'} "
            f"({report.rationale})"
        ],
    }


def _node_revenue_cycle_manager(state: ClaimWorkflowState) -> dict:
    """Terminal orchestration node: the Revenue Cycle Manager reads the
    Compliance Agent's verdict (authoritative), issues a final routing
    decision, and persists the claim's terminal status for this workflow."""
    claim = state["claim"]
    decision = determine_final_route(state)

    assessment = run_revenue_cycle_manager_agent(
        claim, state.get("validation_result"), state.get("insurance_findings"), state.get("compliance_report")
    )
    claim.manager_assessment = assessment
    claim.risk_level = assessment.risk_level
    claim.workflow_stage = "orchestration"

    if decision == "ready_for_submission":
        claim.status = ClaimStatus.READY_FOR_SUBMISSION
        event_type = EventType.CLAIM_READY_FOR_SUBMISSION
    else:
        claim.status = ClaimStatus.NEEDS_HUMAN_REVIEW
        event_type = EventType.HUMAN_REVIEW_REQUIRED

    save_claim(claim)
    save_event(ClaimEvent(claim_id=claim.claim_id, event_type=event_type))

    return {
        "claim": claim,
        "manager_assessment": assessment,
        "route_decision": decision,
        "workflow_stage": "orchestration",
        "logs": [f"Revenue Cycle Manager: risk={assessment.risk_level.value}, "
                 f"priority={assessment.priority_score}, next_step={assessment.next_step}, "
                 f"decision={decision}"],
        "agent_decisions": [f"Revenue Cycle Manager: {assessment.next_step} — {assessment.recommended_action}"],
    }


def build_claim_intake_graph():
    graph = StateGraph(ClaimWorkflowState)

    graph.add_node("clinical_documentation", _node_clinical_documentation)
    graph.add_node("medical_coding", _node_medical_coding)
    graph.add_node("claim_validation", _node_claim_validation)
    graph.add_node("insurance_policy", _node_insurance_policy)
    graph.add_node("compliance", _node_compliance)
    graph.add_node("revenue_cycle_manager", _node_revenue_cycle_manager)

    graph.set_entry_point("clinical_documentation")
    graph.add_edge("clinical_documentation", "medical_coding")
    graph.add_edge("medical_coding", "claim_validation")

    # The feedback loop: Validation can send the claim back to Medical
    # Coding for a bounded number of revisions before proceeding.
    graph.add_conditional_edges(
        "claim_validation",
        route_after_validation,
        {
            "revise_coding": "medical_coding",
            "proceed_to_insurance": "insurance_policy",
        },
    )

    graph.add_edge("insurance_policy", "compliance")
    graph.add_edge("compliance", "revenue_cycle_manager")
    graph.add_edge("revenue_cycle_manager", END)

    # A checkpointer is required for interrupt()/Command(resume=...) to
    # work: it's what lets the Clinical Documentation pause be persisted
    # and resumed later, potentially in a completely different Streamlit
    # rerun, rather than losing all progress. allowed_msgpack_modules=True
    # is safe here because everything checkpointed (Claim, Payer,
    # ClaimStatus, ...) is our own trusted domain model, not external
    # input being deserialized from an untrusted source.
    checkpointer = MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=True))
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Denial / appeal workflow nodes
# ---------------------------------------------------------------------------

def _node_denial_analysis(state: DenialWorkflowState) -> dict:
    claim = state["claim"]
    analysis = run_denial_analysis_agent(claim, claim.denial_reason or "Unspecified denial")
    claim.denial_analysis = analysis
    claim.status = ClaimStatus.DENIED
    claim.workflow_stage = "denial_analysis"
    save_claim(claim)
    save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.CLAIM_DENIED))
    return {
        "claim": claim,
        "denial_analysis": analysis,
        "workflow_stage": "denial_analysis",
        "logs": [f"Denial Analysis Agent: category={analysis.category}, appealable={analysis.is_appealable}"],
    }


def _node_insurance_policy_recheck(state: DenialWorkflowState) -> dict:
    claim = state["claim"]
    findings = run_insurance_policy_agent(claim, claim.clinical_summary)
    claim.insurance_findings = findings
    claim.workflow_stage = "insurance_policy_recheck"
    save_claim(claim)
    return {
        "claim": claim,
        "insurance_findings": findings,
        "workflow_stage": "insurance_policy_recheck",
        "logs": ["Insurance Policy Agent (RAG) re-checked coverage/policy context for the appeal"],
    }


def _node_compliance_denial(state: DenialWorkflowState) -> dict:
    """Compliance quality gate before an appeal is drafted — makes sure
    there's actually a sound basis to appeal on before Appeal Generation
    spends effort drafting a letter."""
    claim = state["claim"]
    report = run_compliance_agent(
        claim,
        claim.clinical_summary,
        claim.coding_result,
        claim.validation_result,
        state.get("insurance_findings"),
        prior_logs=state.get("logs", []),
    )
    claim.compliance_report = report
    claim.workflow_stage = "compliance_review"
    claim.audit_trail = report.audit_trail
    save_claim(claim)

    if not report.is_compliant:
        save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.COMPLIANCE_FAILED))

    return {
        "claim": claim,
        "compliance_report": report,
        "workflow_stage": "compliance_review",
        "logs": [f"Compliance Agent (pre-appeal): is_compliant={report.is_compliant}, "
                 f"human_review_required={report.human_review_required}"],
    }


def _node_appeal_generation(state: DenialWorkflowState) -> dict:
    claim = state["claim"]
    appeal_package = run_appeal_generation_agent(
        claim, state["denial_analysis"], state.get("insurance_findings")
    )
    claim.appeal_package = appeal_package
    claim.status = ClaimStatus.APPEALING
    claim.workflow_stage = "appeal_generation"
    save_claim(claim)

    save_appeal(claim.claim_id, appeal_package)
    save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.APPEAL_SUBMITTED))
    return {
        "claim": claim,
        "appeal_package": appeal_package,
        "workflow_stage": "appeal_generation",
        "logs": [f"Appeal Generation Agent drafted appeal {appeal_package.appeal_id} with "
                 f"{len(appeal_package.missing_documentation_checklist)} checklist items"],
    }


def _node_manager_review(state: DenialWorkflowState) -> dict:
    claim = state["claim"]
    assessment = run_revenue_cycle_manager_agent(
        claim, None, state.get("insurance_findings"), state.get("compliance_report")
    )
    claim.manager_assessment = assessment
    claim.risk_level = assessment.risk_level
    claim.status = ClaimStatus.READY_FOR_RESUBMISSION
    claim.workflow_stage = "orchestration"
    save_claim(claim)
    return {
        "claim": claim,
        "manager_assessment": assessment,
        "route_decision": "ready_for_resubmission",
        "workflow_stage": "orchestration",
        "logs": [f"Revenue Cycle Manager: risk={assessment.risk_level.value}, "
                 f"priority={assessment.priority_score} — ready for resubmission"],
    }


def _node_human_review_required(state: DenialWorkflowState) -> dict:
    claim = state["claim"]
    claim.status = ClaimStatus.NEEDS_HUMAN_REVIEW
    claim.workflow_stage = "human_review"
    save_claim(claim)
    save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.HUMAN_REVIEW_REQUIRED))
    return {
        "claim": claim,
        "route_decision": "needs_human_review",
        "workflow_stage": "human_review",
        "logs": ["Routed to human review (non-appealable denial, or Compliance Agent blocked the appeal)"],
    }


def build_denial_appeal_graph():
    """Insurance Denial -> Denial Analysis -> [appealable?] ->
    Insurance Policy -> Compliance (quality gate) -> [cleared?] ->
    Appeal Generation -> Revenue Cycle Manager -> Ready For Resubmission,
    OR -> Human Review (non-appealable denial, or compliance-blocked)."""
    graph = StateGraph(DenialWorkflowState)

    graph.add_node("denial_analysis", _node_denial_analysis)
    graph.add_node("insurance_policy_recheck", _node_insurance_policy_recheck)
    graph.add_node("compliance_denial", _node_compliance_denial)
    graph.add_node("appeal_generation", _node_appeal_generation)
    graph.add_node("manager_review", _node_manager_review)
    graph.add_node("human_review_required", _node_human_review_required)

    graph.set_entry_point("denial_analysis")
    graph.add_conditional_edges(
        "denial_analysis",
        route_after_denial_analysis,
        {
            "appealable": "insurance_policy_recheck",
            "needs_human_review": "human_review_required",
        },
    )
    graph.add_edge("insurance_policy_recheck", "compliance_denial")
    graph.add_conditional_edges(
        "compliance_denial",
        route_after_compliance_denial,
        {
            "proceed_to_appeal": "appeal_generation",
            "needs_human_review": "human_review_required",
        },
    )
    graph.add_edge("appeal_generation", "manager_review")
    graph.add_edge("manager_review", END)
    graph.add_edge("human_review_required", END)

    return graph.compile()


# Module-level compiled graph singletons (cheap to compile, but avoids
# recompiling — and losing the in-memory checkpointer's state — on every
# Streamlit rerun).
_claim_intake_graph = None
_denial_appeal_graph = None


def get_claim_intake_graph():
    global _claim_intake_graph
    if _claim_intake_graph is None:
        _claim_intake_graph = build_claim_intake_graph()
    return _claim_intake_graph


def get_denial_appeal_graph():
    global _denial_appeal_graph
    if _denial_appeal_graph is None:
        _denial_appeal_graph = build_denial_appeal_graph()
    return _denial_appeal_graph
