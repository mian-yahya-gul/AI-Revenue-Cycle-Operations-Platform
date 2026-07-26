"""Agent 3 — Claim Validation Agent.

Wraps the rule-based validation tool as an agent step (kept rule-based
even in live-LLM mode, since deterministic claim-edit logic like
duplicate detection and required-field checks should not be
non-deterministic in a production billing platform).
"""

from __future__ import annotations

import time

from database.repository import log_agent_action
from models.schemas import AgentLogEntry, Claim, CodingResult, ValidationResult
from tools.validation_tools import run_claim_validation
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Claim Validation Agent"


def run_claim_validation_agent(claim: Claim, coding_result: CodingResult | None) -> ValidationResult:
    start = time.time()
    logger.info("[%s] Processing claim %s", AGENT_NAME, claim.claim_id)

    result = run_claim_validation(claim, coding_result)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim.claim_id,
            agent_name=AGENT_NAME,
            action="validate_claim",
            input_summary=f"billed_amount={claim.billed_amount}, payer={claim.payer.value}",
            output_summary=f"is_valid={result.is_valid}, completeness={result.completeness_score}, "
            f"issues={len(result.issues)}",
            duration_ms=duration_ms,
        )
    )
    return result
