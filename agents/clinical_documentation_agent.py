"""Agent 1 — Clinical Documentation Agent.

Reads raw physician notes and produces a structured ClinicalSummary.
"""

from __future__ import annotations

import re
import time

from config.settings import settings
from database.repository import log_agent_action
from models.schemas import AgentLogEntry, ClinicalSummary
from prompts.agent_prompts import CLINICAL_DOCUMENTATION_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Clinical Documentation Agent"

_DIAGNOSIS_KEYWORDS = [
    "chest pain", "type 2 diabetes", "diabetes mellitus", "hypertension",
    "bronchitis", "low back pain", "lower back pain", "appendicitis",
    "pneumonia", "atrial fibrillation", "afib", "acute kidney injury",
    "migraine", "gastroenteritis", "wrist fracture", "fracture of wrist",
    "fracture, right", "fracture, left", "asthma",
    "urinary tract infection", "uti", "concussion",
]
_DIAGNOSIS_CANONICAL = {
    "fracture of wrist": "wrist fracture",
    "fracture, right": "wrist fracture",
    "fracture, left": "wrist fracture",
}
_PROCEDURE_KEYWORDS = [
    "x-ray", "ct scan", "ct chest", "ct abdomen", "mri", "ekg",
    "electrocardiogram", "blood count", "metabolic panel", "appendectomy",
    "nebulizer", "urinalysis", "office visit", "emergency department visit",
    "closed treatment",
]


def _contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word/phrase match on whitespace-normalized text, avoiding
    false positives like 'uti' inside 'routine' and false negatives from
    a phrase being split across a line wrap."""
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text) is not None


def _mock_extract(notes: str) -> ClinicalSummary:
    """Deterministic keyword-based extraction used when no LLM is configured."""
    text = re.sub(r"\s+", " ", notes.lower())

    diagnoses = sorted(
        {_DIAGNOSIS_CANONICAL.get(kw, kw) for kw in _DIAGNOSIS_KEYWORDS if _contains_phrase(text, kw)}
    )
    procedures = sorted({kw for kw in _PROCEDURE_KEYWORDS if _contains_phrase(text, kw)})

    physician_match = re.search(r"(?:Dr\.?|Physician:)\s*([A-Z][a-zA-Z\.\-]+(?:\s[A-Z][a-zA-Z\.\-]+)?)", notes)
    physician_name = physician_match.group(1).strip() if physician_match else None

    date_match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})", notes)
    date_of_service = date_match.group(1) if date_match else None

    chief_complaint = None
    cc_match = re.search(r"chief complaint:?\s*(.+)", notes, re.IGNORECASE)
    if cc_match:
        chief_complaint = cc_match.group(1).split("\n")[0].strip()

    missing_information = []
    if not physician_name:
        missing_information.append("Physician name/signature not clearly identified")
    if not date_of_service:
        missing_information.append("Date of service not clearly identified")
    if not diagnoses:
        missing_information.append("No recognizable diagnosis documented")
    if not procedures:
        missing_information.append("No recognizable procedure documented")
    if "signed" not in text and "signature" not in text:
        missing_information.append("Note does not appear to reference a physician signature")

    confidence = 0.9 if (diagnoses and procedures and physician_name) else 0.6

    summary_text = (
        f"Patient presented with findings consistent with: {', '.join(diagnoses) or 'unclear diagnosis'}. "
        f"Documented services/procedures: {', '.join(procedures) or 'none clearly documented'}."
    )

    return ClinicalSummary(
        diagnoses=diagnoses,
        procedures=procedures,
        chief_complaint=chief_complaint,
        physician_name=physician_name,
        date_of_service=date_of_service,
        missing_information=missing_information,
        summary_text=summary_text,
        confidence=confidence,
    )


def _llm_extract(notes: str) -> ClinicalSummary:
    from utils.llm_client import get_structured_llm

    structured_llm = get_structured_llm(ClinicalSummary)
    response = structured_llm.invoke(
        [
            {"role": "system", "content": CLINICAL_DOCUMENTATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Physician notes:\n\n{notes}"},
        ]
    )
    return response


def run_clinical_documentation_agent(claim_id: str, physician_notes: str) -> ClinicalSummary:
    """Entry point used by the LangGraph node."""
    start = time.time()
    logger.info("[%s] Processing claim %s", AGENT_NAME, claim_id)

    if settings.use_mock_llm:
        result = _mock_extract(physician_notes)
    else:
        try:
            result = _llm_extract(physician_notes)
        except Exception:
            logger.exception("[%s] LLM extraction failed, falling back to mock", AGENT_NAME)
            result = _mock_extract(physician_notes)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim_id,
            agent_name=AGENT_NAME,
            action="extract_clinical_summary",
            input_summary=physician_notes[:200],
            output_summary=result.summary_text[:200],
            duration_ms=duration_ms,
        )
    )
    return result
