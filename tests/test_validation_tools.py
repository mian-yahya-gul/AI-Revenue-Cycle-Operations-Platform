"""Tests for the rule-based claim validation tool."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Point the app at an isolated temp database before importing anything
# that touches config/settings at import time.
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ["DATABASE_PATH"] = _TMP_DB
os.environ["USE_MOCK_LLM"] = "true"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db import init_db, reset_db  # noqa: E402
from database.repository import save_claim  # noqa: E402
from models.schemas import (  # noqa: E402
    Claim,
    CodeRecommendation,
    CodingResult,
    Payer,
)
from tools.validation_tools import run_claim_validation  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db():
    reset_db()
    init_db()
    yield


def _make_claim(**overrides) -> Claim:
    defaults = dict(
        patient_id="PT-TEST0001",
        patient_name="Test Patient",
        payer=Payer.AETNA,
        date_of_service="2026-06-01",
        billed_amount=500.0,
        physician_notes_raw="Test notes",
    )
    defaults.update(overrides)
    return Claim(**defaults)


def _make_coding(confidence: float = 0.9, with_codes: bool = True) -> CodingResult:
    if not with_codes:
        return CodingResult(icd10_codes=[], cpt_codes=[], overall_confidence=0.0)
    return CodingResult(
        icd10_codes=[CodeRecommendation(code="R07.9", description="Chest pain", reasoning="x", confidence=confidence)],
        cpt_codes=[CodeRecommendation(code="99213", description="Office visit", reasoning="x", confidence=confidence)],
        overall_confidence=confidence,
    )


def test_valid_claim_passes_validation():
    claim = _make_claim()
    coding = _make_coding(confidence=0.9)
    result = run_claim_validation(claim, coding)
    assert result.is_valid is True
    assert result.duplicate_claim_detected is False
    assert result.completeness_score == 1.0


def test_missing_coding_marks_invalid():
    claim = _make_claim()
    coding = _make_coding(with_codes=False)
    result = run_claim_validation(claim, coding)
    assert result.is_valid is False
    fields = {issue.field for issue in result.issues}
    assert "icd10_codes" in fields
    assert "cpt_codes" in fields


def test_zero_billed_amount_is_critical():
    claim = _make_claim(billed_amount=0.0)
    coding = _make_coding()
    result = run_claim_validation(claim, coding)
    assert result.is_valid is False
    assert any(i.field == "billed_amount" for i in result.issues)


def test_duplicate_claim_detected():
    first = _make_claim(date_of_service="2026-07-01")
    save_claim(first)

    second = _make_claim(date_of_service="2026-07-01")
    coding = _make_coding()
    result = run_claim_validation(second, coding)

    assert result.duplicate_claim_detected is True
    assert any("duplicate" in i.issue.lower() for i in result.issues)


def test_low_confidence_coding_flags_warning():
    claim = _make_claim()
    coding = _make_coding(confidence=0.3)
    result = run_claim_validation(claim, coding)
    warning_issues = [i for i in result.issues if i.severity == "warning"]
    assert any("confidence" in i.issue.lower() for i in warning_issues)
