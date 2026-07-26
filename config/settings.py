"""
Application-wide configuration.

All runtime configuration is sourced from environment variables (via a
.env file in local development) so the platform behaves consistently
across local, CI, and containerized deployments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
PAYER_POLICY_DIR = SAMPLE_DATA_DIR / "payer_policies"
PHYSICIAN_NOTES_DIR = SAMPLE_DATA_DIR / "physician_notes"
DENIALS_DIR = SAMPLE_DATA_DIR / "denials"
VECTOR_STORE_DIR = BASE_DIR / "rag" / "vector_store_index"
LOG_DIR = BASE_DIR / "logs"
DB_DIR = BASE_DIR / "database"


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Centralized, immutable application settings."""

    # --- LLM Provider ---------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Mock mode --------------------------------------------------------
    # When no API key is configured (e.g. a recruiter cloning the repo),
    # the platform falls back to deterministic rule-based agent logic
    # instead of calling a live LLM, so the whole workflow still runs.
    use_mock_llm: bool = field(
        default_factory=lambda: _get_bool("USE_MOCK_LLM", os.getenv("OPENAI_API_KEY", "") == "")
    )

    # --- Database ---------------------------------------------------------
    database_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", str(DB_DIR / "revenue_cycle.db"))
    )

    # --- RAG ---------------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    retriever_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_TOP_K", "4")))

    # --- Workflow thresholds ------------------------------------------------
    validation_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("VALIDATION_CONFIDENCE_THRESHOLD", "0.75"))
    )
    high_risk_claim_amount: float = field(
        default_factory=lambda: float(os.getenv("HIGH_RISK_CLAIM_AMOUNT", "5000"))
    )

    # --- Multi-agent collaboration -----------------------------------------
    # Maximum number of Coding <-> Validation feedback-loop revision cycles
    # before the claim is forced downstream to the Compliance Agent /
    # Revenue Cycle Manager for escalation rather than looping forever.
    max_coding_revisions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CODING_REVISIONS", "2"))
    )
    # If the Clinical Documentation Agent extracts nothing usable at all
    # (no diagnoses AND no procedures), the intake workflow pauses via a
    # LangGraph interrupt() and waits for a human to supply additional
    # documentation, rather than pushing an empty claim further downstream.
    documentation_pause_enabled: bool = field(
        default_factory=lambda: _get_bool("DOCUMENTATION_PAUSE_ENABLED", True)
    )

    # --- App metadata --------------------------------------------------------
    app_name: str = "AI Medical Billing Revenue Cycle Team"
    app_version: str = "1.1.0"
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()

for _dir in (LOG_DIR, VECTOR_STORE_DIR, DB_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
