"""Conditional routing functions used as LangGraph edge selectors.

These are where the "team of specialists" behavior actually lives:
whether the Medical Coding Agent gets called back in for a revision,
whether a denial proceeds to appeal, and what the Revenue Cycle Manager's
final orchestration decision is all get decided here, based on what
upstream agents wrote into shared state — never by one agent calling
another directly.
"""

from __future__ import annotations

from config.settings import settings
from graph.state import ClaimWorkflowState, DenialWorkflowState
from utils.logger import get_logger

logger = get_logger(__name__)


def _coding_needs_revision(state: ClaimWorkflowState) -> tuple[bool, str]:
    """Decide whether the Medical Coding Agent's output is good enough to
    proceed, or whether Claim Validation should send it back for revision.
    Returns (needs_revision, feedback_message)."""
    coding = state.get("coding_result")
    if coding is None:
        return False, ""

    problems = []
    if not coding.icd10_codes:
        problems.append("no ICD-10 diagnosis codes were assigned")
    if not coding.cpt_codes:
        problems.append("no CPT procedure codes were assigned")
    if coding.overall_confidence < settings.validation_confidence_threshold:
        problems.append(f"overall coding confidence ({coding.overall_confidence:.2f}) is below "
                         f"the {settings.validation_confidence_threshold} threshold")

    if not problems:
        return False, ""
    return True, "; ".join(problems)


def route_after_validation(state: ClaimWorkflowState) -> str:
    """Coding <-> Validation feedback loop control.

    If Claim Validation finds coding-related problems and the claim
    hasn't already exhausted its revision budget, route back to the
    Medical Coding Agent with feedback for a revision, then validate
    again. Otherwise proceed to the Insurance Policy Agent.
    """
    revision_count = state.get("revision_count", 0)
    needs_revision, feedback = _coding_needs_revision(state)

    if needs_revision and revision_count < settings.max_coding_revisions:
        logger.info(
            "Claim %s: Validation is sending coding back for revision (%d/%d): %s",
            state["claim"].claim_id, revision_count + 1, settings.max_coding_revisions, feedback,
        )
        return "revise_coding"

    if needs_revision:
        logger.info(
            "Claim %s: coding issues remain after %d revision attempt(s) — proceeding "
            "to Insurance Policy; Compliance Agent will flag for human review.",
            state["claim"].claim_id, revision_count,
        )
    return "proceed_to_insurance"


def determine_final_route(state: ClaimWorkflowState) -> str:
    """Final routing decision for the claim-intake workflow, made by the
    Revenue Cycle Manager. The Compliance Agent's human_review_required
    flag is authoritative; validation/insurance checks are a defensive
    fallback in case a compliance report is unexpectedly absent.
    """
    compliance = state.get("compliance_report")
    if compliance is not None:
        decision = "needs_human_review" if compliance.human_review_required else "ready_for_submission"
        logger.info("Routing decision for claim %s: %s (compliance-gated)", state["claim"].claim_id, decision)
        return decision

    # Defensive fallback — should not normally be reached, since the
    # Compliance Agent always runs before this node.
    validation = state.get("validation_result")
    insurance = state.get("insurance_findings")
    needs_review = bool(
        (validation and not validation.is_valid)
        or (validation and validation.duplicate_claim_detected)
        or (insurance and insurance.prior_authorization_required)
        or (insurance and not insurance.is_covered)
    )
    decision = "needs_human_review" if needs_review else "ready_for_submission"
    logger.warning(
        "Routing decision for claim %s: %s (fallback — no compliance report present)",
        state["claim"].claim_id, decision,
    )
    return decision


def route_after_denial_analysis(state: DenialWorkflowState) -> str:
    """Decide whether a denied claim proceeds to appeal generation or is
    routed straight to human review (non-appealable denials)."""
    denial = state.get("denial_analysis")
    if denial and not denial.is_appealable:
        logger.info("Denial for claim %s is not appealable — routing to human review", state["claim"].claim_id)
        return "needs_human_review"
    return "appealable"


def route_after_compliance_denial(state: DenialWorkflowState) -> str:
    """After the Compliance Agent reviews a denial's appeal readiness,
    decide whether to proceed to Appeal Generation or escalate straight
    to human review (e.g. documentation is still fundamentally missing)."""
    compliance = state.get("compliance_report")
    if compliance and compliance.human_review_required:
        logger.info(
            "Claim %s: Compliance Agent blocked appeal generation — routing to human review",
            state["claim"].claim_id,
        )
        return "needs_human_review"
    return "proceed_to_appeal"
