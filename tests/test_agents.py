"""Tests for the mock-mode Clinical Documentation and Medical Coding agents."""

from __future__ import annotations

import os
import sys
import tempfile

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ["DATABASE_PATH"] = _TMP_DB
os.environ["USE_MOCK_LLM"] = "true"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

from agents.clinical_documentation_agent import run_clinical_documentation_agent  # noqa: E402
from agents.medical_coding_agent import run_medical_coding_agent  # noqa: E402
from database.db import init_db, reset_db  # noqa: E402

CHEST_PAIN_NOTE = """\
Chief Complaint: Chest pain

HPI: Patient presents with acute chest pain.

Assessment: Chest pain, unspecified.

Plan: EKG obtained, chest x-ray obtained.

Physician: Dr. Sarah Chen
Signed and dated by attending physician.
"""

INCOMPLETE_NOTE = "Patient seen today. Doing okay. Continue current plan."


@pytest.fixture(autouse=True)
def _fresh_db():
    reset_db()
    init_db()
    yield


def test_clinical_documentation_extracts_diagnosis_and_procedure():
    summary = run_clinical_documentation_agent("CLM-TEST01", CHEST_PAIN_NOTE)
    assert "chest pain" in summary.diagnoses
    assert "ekg" in summary.procedures
    assert "x-ray" in summary.procedures
    assert summary.physician_name is not None
    assert summary.confidence > 0.5


def test_clinical_documentation_flags_missing_info_on_sparse_note():
    summary = run_clinical_documentation_agent("CLM-TEST02", INCOMPLETE_NOTE)
    assert len(summary.missing_information) > 0
    assert summary.diagnoses == []


def test_medical_coding_assigns_codes_for_recognized_terms():
    summary = run_clinical_documentation_agent("CLM-TEST03", CHEST_PAIN_NOTE)
    coding = run_medical_coding_agent("CLM-TEST03", summary)
    assert any(c.code == "R07.9" for c in coding.icd10_codes)
    assert any(c.code == "93000" for c in coding.cpt_codes)
    assert coding.overall_confidence > 0.5


def test_medical_coding_low_confidence_when_no_diagnosis():
    summary = run_clinical_documentation_agent("CLM-TEST04", INCOMPLETE_NOTE)
    coding = run_medical_coding_agent("CLM-TEST04", summary)
    assert coding.icd10_codes == []
    assert coding.overall_confidence == 0.0
