"""ICD-10 and CPT lookup tools (mock reference data)."""

from __future__ import annotations

from langchain_core.tools import tool

from tools.coding_reference import lookup_cpt, lookup_icd10


@tool("icd_lookup_tool")
def icd_lookup_tool(clinical_term: str) -> str:
    """Look up the most likely ICD-10 diagnosis code for a clinical term or
    diagnosis phrase (e.g. 'chest pain', 'type 2 diabetes')."""
    result = lookup_icd10(clinical_term)
    if not result:
        return f"No ICD-10 match found for '{clinical_term}'. Manual coding review required."
    return f"{result['code']} — {result['description']}"


@tool("cpt_lookup_tool")
def cpt_lookup_tool(procedure_term: str) -> str:
    """Look up the most likely CPT procedure code for a procedure or
    service description (e.g. 'chest x-ray', 'office visit level 4')."""
    result = lookup_cpt(procedure_term)
    if not result:
        return f"No CPT match found for '{procedure_term}'. Manual coding review required."
    return f"{result['code']} — {result['description']}"
