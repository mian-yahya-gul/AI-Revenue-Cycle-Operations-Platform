"""Agent 2 — Medical Coding Agent.

Recommends ICD-10 and CPT codes from the structured clinical summary.
"""

from __future__ import annotations

import time

from config.settings import settings
from database.repository import log_agent_action
from models.schemas import AgentLogEntry, ClinicalSummary, CodeRecommendation, CodingResult
from prompts.agent_prompts import MEDICAL_CODING_SYSTEM_PROMPT
from tools.coding_reference import lookup_cpt, lookup_icd10
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Medical Coding Agent"


def _mock_code(clinical_summary: ClinicalSummary, feedback: str | None = None) -> CodingResult:
    icd_codes: list[CodeRecommendation] = []
    for diagnosis in clinical_summary.diagnoses:
        match = lookup_icd10(diagnosis)
        if match:
            icd_codes.append(
                CodeRecommendation(
                    code=match["code"],
                    description=match["description"],
                    reasoning=f"Directly supported by documented diagnosis: '{diagnosis}'",
                    confidence=0.9,
                )
            )

    cpt_codes: list[CodeRecommendation] = []
    for procedure in clinical_summary.procedures:
        match = lookup_cpt(procedure)
        if match:
            cpt_codes.append(
                CodeRecommendation(
                    code=match["code"],
                    description=match["description"],
                    reasoning=f"Directly supported by documented procedure: '{procedure}'",
                    confidence=0.88,
                )
            )

    revised = False
    if feedback:
        # This is a revision pass triggered by the Claim Validation Agent.
        # Behave like a colleague responding to specific feedback rather
        # than blindly repeating the same lookup: attempt a conservative,
        # clearly-flagged fallback for a missing CPT code when diagnoses
        # are documented but no procedure could be matched. We do NOT
        # fabricate a diagnosis code — inventing a clinical fact the
        # documentation doesn't support is a compliance risk, not a fix —
        # so an empty diagnosis list is correctly left for human coding.
        if icd_codes and not cpt_codes:
            fallback = lookup_cpt("office visit level 3")
            if fallback:
                cpt_codes.append(
                    CodeRecommendation(
                        code=fallback["code"],
                        description=fallback["description"],
                        reasoning="Fallback evaluation-and-management code applied during coding "
                        "revision after Claim Validation flagged a missing CPT code. Diagnoses "
                        "were documented but no specific procedure was identified — this default "
                        "requires manual confirmation before submission.",
                        confidence=0.5,
                    )
                )
                revised = True

    if not icd_codes:
        overall_confidence = 0.0
        notes = "No ICD-10 codes could be matched from the clinical summary. Manual coding required."
    elif not cpt_codes:
        overall_confidence = 0.4
        notes = "Diagnosis codes matched, but no CPT procedure codes could be matched. Manual review recommended."
    else:
        confidences = [c.confidence for c in icd_codes + cpt_codes]
        overall_confidence = round(sum(confidences) / len(confidences), 2)
        notes = "Codes assigned based on direct keyword match against documented diagnoses/procedures."

    if feedback:
        notes = (
            f"Revision requested by Claim Validation Agent: {feedback}. "
            + ("Applied a fallback CPT code pending manual confirmation. " if revised else
               "No safe automated fix was available for the flagged issue(s); escalating. ")
            + notes
        )

    return CodingResult(
        icd10_codes=icd_codes,
        cpt_codes=cpt_codes,
        overall_confidence=overall_confidence,
        coding_notes=notes,
    )


def _llm_code(clinical_summary: ClinicalSummary, feedback: str | None = None) -> CodingResult:
    from utils.llm_client import get_structured_llm

    structured_llm = get_structured_llm(CodingResult)
    user_content = (
        f"Clinical summary:\n{clinical_summary.summary_text}\n\n"
        f"Diagnoses: {clinical_summary.diagnoses}\n"
        f"Procedures: {clinical_summary.procedures}\n"
        f"Missing information: {clinical_summary.missing_information}"
    )
    if feedback:
        user_content += (
            f"\n\nThis is a revision. The Claim Validation Agent flagged the following on "
            f"your previous coding attempt — address it directly: {feedback}"
        )

    response = structured_llm.invoke(
        [
            {"role": "system", "content": MEDICAL_CODING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    return response


def run_medical_coding_agent(
    claim_id: str, clinical_summary: ClinicalSummary, feedback: str | None = None
) -> CodingResult:
    start = time.time()
    if feedback:
        logger.info("[%s] Revising coding for claim %s based on validation feedback", AGENT_NAME, claim_id)
    else:
        logger.info("[%s] Processing claim %s", AGENT_NAME, claim_id)

    if settings.use_mock_llm:
        result = _mock_code(clinical_summary, feedback)
    else:
        try:
            result = _llm_code(clinical_summary, feedback)
        except Exception:
            logger.exception("[%s] LLM coding failed, falling back to mock", AGENT_NAME)
            result = _mock_code(clinical_summary, feedback)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim_id,
            agent_name=AGENT_NAME,
            action="revise_codes" if feedback else "recommend_codes",
            input_summary=f"diagnoses={clinical_summary.diagnoses}, procedures={clinical_summary.procedures}"
            + (f", feedback={feedback}" if feedback else ""),
            output_summary=f"icd10={[c.code for c in result.icd10_codes]}, cpt={[c.code for c in result.cpt_codes]}",
            duration_ms=duration_ms,
        )
    )
    return result
