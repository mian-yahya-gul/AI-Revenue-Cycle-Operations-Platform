"""
Event-driven workflow entry points.

Each public function here corresponds to a business event (New Claim,
Insurance Denial, Documentation Uploaded, etc). Firing the event persists
it, then triggers only the relevant LangGraph workflow (or resumes a
paused one) — this is the "event-driven" layer sitting above the
orchestration graphs, and is what the Streamlit UI and any future API
layer should call rather than invoking the graphs directly.
"""

from __future__ import annotations

from langgraph.types import Command

from database.repository import save_claim, save_event
from graph.graph import get_claim_intake_graph, get_denial_appeal_graph
from models.schemas import Claim, ClaimEvent, ClaimStatus, EventType
from utils.logger import get_logger

logger = get_logger(__name__)


def _thread_config(claim_id: str) -> dict:
    """Every claim gets its own LangGraph checkpoint thread, keyed by
    claim_id, so pausing/resuming one claim's workflow never touches
    another's in-flight state."""
    return {"configurable": {"thread_id": claim_id}}


def submit_new_claim(claim: Claim) -> Claim:
    """Fires the NEW_CLAIM event and runs the claim intake workflow.

    If the Clinical Documentation Agent finds the notes too sparse to
    proceed, the graph pauses mid-run (see graph/graph.py) and this
    returns the claim in AWAITING_DOCUMENTATION status rather than a
    fully-processed result — call resume_claim_with_documentation() later
    to continue it.
    """
    save_claim(claim)
    save_event(ClaimEvent(claim_id=claim.claim_id, event_type=EventType.NEW_CLAIM))
    logger.info("New claim event fired for %s", claim.claim_id)

    graph = get_claim_intake_graph()
    config = _thread_config(claim.claim_id)
    result = graph.invoke({"claim": claim, "logs": [], "revision_count": 0}, config=config)

    if "__interrupt__" in result:
        logger.info("Claim %s workflow paused awaiting documentation", claim.claim_id)
        from database.repository import get_claim

        return get_claim(claim.claim_id)

    return result["claim"]


def resume_claim_with_documentation(claim_id: str, updated_notes: str) -> Claim:
    """Fires DOCUMENT_UPDATED and resumes a claim paused on the
    documentation interrupt, continuing the graph from exactly where it
    left off — the Clinical Documentation Agent re-runs against the
    updated notes, and every downstream agent that depends on its output
    runs fresh, since they're all "affected" by the correction.
    """
    from database.repository import get_claim

    claim = get_claim(claim_id)
    if claim is None:
        raise ValueError(f"Cannot resume: no claim found with ID {claim_id}")
    if claim.status != ClaimStatus.AWAITING_DOCUMENTATION:
        raise ValueError(
            f"Claim {claim_id} is not awaiting documentation (status={claim.status.value}); nothing to resume."
        )

    graph = get_claim_intake_graph()
    config = _thread_config(claim_id)
    result = graph.invoke(Command(resume={"updated_notes": updated_notes}), config=config)

    if "__interrupt__" in result:
        logger.info("Claim %s paused again — still insufficient documentation", claim_id)
        return get_claim(claim_id)

    return result["claim"]


def process_denial(claim_id: str, denial_reason: str) -> Claim:
    """Fires the CLAIM_DENIED event and runs the denial/appeal workflow."""
    from database.repository import get_claim

    claim = get_claim(claim_id)
    if claim is None:
        raise ValueError(f"Cannot process denial: no claim found with ID {claim_id}")

    claim.denial_reason = denial_reason
    claim.status = ClaimStatus.DENIED
    save_claim(claim)

    graph = get_denial_appeal_graph()
    final_state = graph.invoke({"claim": claim, "logs": []})
    return final_state["claim"]
