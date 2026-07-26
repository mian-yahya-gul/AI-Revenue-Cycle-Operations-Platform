"""Agent 4 — Insurance Policy Agent (RAG).

Uses the NumPy cosine-similarity retriever over payer policy documents to determine
coverage, prior authorization requirements, and required documentation.
This agent must ground its answer in retrieved text rather than model
memory, since payer policy specifics vary and change over time.
"""

from __future__ import annotations

import time

from config.settings import settings
from database.repository import log_agent_action
from models.schemas import AgentLogEntry, Claim, ClinicalSummary, InsuranceFindings, PolicyReference
from prompts.agent_prompts import INSURANCE_POLICY_SYSTEM_PROMPT
from rag.retriever import format_chunks_for_prompt, retrieve_policy_chunks
from utils.logger import get_logger

logger = get_logger(__name__)

AGENT_NAME = "Insurance Policy Agent"

_PRIOR_AUTH_SERVICE_HINTS = [
    "mri", "ct", "ct scan", "pet", "inpatient", "surgery", "surgical",
    "durable medical equipment", "dme", "infusion", "genetic",
]


def _build_query(claim: Claim, clinical_summary: ClinicalSummary | None) -> str:
    procedures = ", ".join(clinical_summary.procedures) if clinical_summary else ""
    diagnoses = ", ".join(clinical_summary.diagnoses) if clinical_summary else ""
    return (
        f"prior authorization and coverage requirements for procedures: {procedures} "
        f"with diagnoses: {diagnoses}"
    )


def _detect_conflicts(chunks, prior_auth_required: bool) -> list[str]:
    """Light-touch heuristic conflict detection across retrieved chunks.

    Only flags a conflict when there's an explicit contradiction in the
    retrieved text — e.g. a coverage-criteria document explicitly states
    a service does *not* require prior authorization while our own
    procedure-based determination (or the prior-authorization document)
    says it does. Two different document *types* being retrieved
    together is normal and NOT itself a conflict — most claims will
    correctly show zero conflicts; this exists to catch real
    contradictions when they occur, not to manufacture false ones.
    """
    conflicts = []
    by_doc_type: dict[str, list] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        doc_type = "prior_authorization" if "prior_authorization" in source else (
            "coverage_criteria" if "coverage_criteria" in source else "other"
        )
        by_doc_type.setdefault(doc_type, []).append(chunk)

    if prior_auth_required and "coverage_criteria" in by_doc_type:
        cov_text = " ".join(c.page_content.lower() for c in by_doc_type["coverage_criteria"])
        negation_phrases = ["does not require prior authorization", "no prior authorization",
                             "without prior authorization", "prior authorization is not required"]
        if any(phrase in cov_text for phrase in negation_phrases):
            conflicts.append(
                "This claim's procedure category was matched as requiring prior authorization, "
                "but the retrieved coverage criteria document explicitly states prior "
                "authorization is not required — confirm which rule applies before submission."
            )
    return conflicts


def _mock_analyze(claim: Claim, clinical_summary: ClinicalSummary | None, chunks) -> InsuranceFindings:
    procedures_text = " ".join(clinical_summary.procedures if clinical_summary else []).lower()
    prior_auth_required = any(hint in procedures_text for hint in _PRIOR_AUTH_SERVICE_HINTS)

    required_documents = ["Signed physician order", "Clinical notes supporting medical necessity"]
    if prior_auth_required:
        required_documents.append("Prior authorization / precertification approval number")

    policy_refs = [
        PolicyReference(
            payer=chunk.metadata.get("payer", claim.payer.value),
            document=chunk.metadata.get("source", "unknown"),
            excerpt=chunk.page_content.strip()[:280],
        )
        for chunk in chunks[:2]
    ]

    conflicting_rules = _detect_conflicts(chunks, prior_auth_required)

    matched_hints = [h for h in _PRIOR_AUTH_SERVICE_HINTS if h in procedures_text]
    if prior_auth_required:
        rationale = (
            f"Documented procedure(s) ({', '.join(clinical_summary.procedures) if clinical_summary else 'none'}) "
            f"match service categories ({', '.join(matched_hints)}) that {claim.payer.value}'s policy documents "
            f"list as requiring prior authorization."
        )
    else:
        rationale = (
            f"Documented procedure(s) did not match any {claim.payer.value} service category that the "
            f"retrieved policy excerpts list as requiring prior authorization."
        )

    notes = (
        f"Based on {claim.payer.value} policy documents, this claim "
        f"{'appears to require prior authorization' if prior_auth_required else 'does not clearly require prior authorization'} "
        "based on the procedures documented. Coverage assumed valid pending "
        "standard medical necessity documentation."
    )

    return InsuranceFindings(
        is_covered=True,
        prior_authorization_required=prior_auth_required,
        required_documents=required_documents,
        policy_references=policy_refs,
        rationale=rationale,
        conflicting_rules=conflicting_rules,
        notes=notes,
    )


def _llm_analyze(claim: Claim, clinical_summary: ClinicalSummary | None, chunks) -> InsuranceFindings:
    from utils.llm_client import get_structured_llm

    structured_llm = get_structured_llm(InsuranceFindings)
    context = format_chunks_for_prompt(chunks)
    procedures = ", ".join(clinical_summary.procedures) if clinical_summary else "unknown"
    diagnoses = ", ".join(clinical_summary.diagnoses) if clinical_summary else "unknown"

    response = structured_llm.invoke(
        [
            {"role": "system", "content": INSURANCE_POLICY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Payer: {claim.payer.value}\n"
                    f"Diagnoses: {diagnoses}\n"
                    f"Procedures: {procedures}\n\n"
                    f"Retrieved policy excerpts:\n{context}"
                ),
            },
        ]
    )
    return response


def run_insurance_policy_agent(claim: Claim, clinical_summary: ClinicalSummary | None) -> InsuranceFindings:
    start = time.time()
    logger.info("[%s] Processing claim %s", AGENT_NAME, claim.claim_id)

    query = _build_query(claim, clinical_summary)
    chunks = retrieve_policy_chunks(query, payer=claim.payer.value)

    if settings.use_mock_llm:
        result = _mock_analyze(claim, clinical_summary, chunks)
    else:
        try:
            result = _llm_analyze(claim, clinical_summary, chunks)
        except Exception:
            logger.exception("[%s] LLM analysis failed, falling back to mock", AGENT_NAME)
            result = _mock_analyze(claim, clinical_summary, chunks)

    duration_ms = int((time.time() - start) * 1000)
    log_agent_action(
        AgentLogEntry(
            claim_id=claim.claim_id,
            agent_name=AGENT_NAME,
            action="rag_policy_lookup",
            input_summary=query[:200],
            output_summary=f"covered={result.is_covered}, prior_auth={result.prior_authorization_required}, "
            f"sources={len(result.policy_references)}, conflicts={len(result.conflicting_rules)}",
            duration_ms=duration_ms,
        )
    )
    return result
