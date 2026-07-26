"""SQLite connection management and schema creation."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL DEFAULT 'billing_specialist',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    payer TEXT NOT NULL,
    member_id TEXT NOT NULL,
    plan_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    payer TEXT NOT NULL,
    date_of_service TEXT NOT NULL,
    billed_amount REAL NOT NULL,
    physician_notes_raw TEXT NOT NULL,
    status TEXT NOT NULL,
    risk_level TEXT NOT NULL DEFAULT 'low',
    denial_reason TEXT,
    clinical_summary_json TEXT,
    coding_result_json TEXT,
    validation_result_json TEXT,
    insurance_findings_json TEXT,
    compliance_report_json TEXT,
    denial_analysis_json TEXT,
    appeal_package_json TEXT,
    manager_assessment_json TEXT,
    workflow_stage TEXT NOT NULL DEFAULT 'intake',
    revision_count INTEGER NOT NULL DEFAULT 0,
    audit_trail_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    payer TEXT NOT NULL,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS appeals (
    appeal_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    appeal_letter TEXT NOT NULL,
    missing_documentation_checklist TEXT,
    supporting_evidence_summary TEXT,
    status TEXT NOT NULL DEFAULT 'drafted',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_logs (
    log_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    input_summary TEXT,
    output_summary TEXT,
    duration_ms INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_risk ON claims(risk_level);
CREATE INDEX IF NOT EXISTS idx_agent_logs_claim ON agent_logs(claim_id);
CREATE INDEX IF NOT EXISTS idx_events_claim ON events(claim_id);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.database_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    """Context manager yielding a connection with commit/rollback handling."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database transaction rolled back")
        raise
    finally:
        conn.close()


# Columns added after the initial release. CREATE TABLE IF NOT EXISTS is a
# no-op against an existing table, so anyone with a database file created
# before these columns existed would otherwise hit "no such column" errors
# the first time the new multi-agent-collaboration fields are written. This
# migration adds any missing columns in place, preserving existing data.
_CLAIMS_TABLE_MIGRATIONS: dict[str, str] = {
    "compliance_report_json": "TEXT",
    "workflow_stage": "TEXT NOT NULL DEFAULT 'intake'",
    "revision_count": "INTEGER NOT NULL DEFAULT 0",
    "audit_trail_json": "TEXT",
}


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(claims)").fetchall()}
    for column, ddl_type in _CLAIMS_TABLE_MIGRATIONS.items():
        if column not in existing_columns:
            logger.info("Migrating claims table: adding column %s", column)
            conn.execute(f"ALTER TABLE claims ADD COLUMN {column} {ddl_type}")


def init_db() -> None:
    """Create all tables if they do not already exist, and migrate any
    existing database in place to add columns introduced by later
    architecture changes."""
    with db_session() as conn:
        conn.executescript(SCHEMA)
        _migrate_schema(conn)
    logger.info("Database initialized at %s", settings.database_path)


def reset_db() -> None:
    """Drop and recreate all tables. Used by the sample-data seeding script."""
    tables = [
        "agent_logs",
        "events",
        "appeals",
        "claims",
        "policies",
        "patients",
        "users",
    ]
    with db_session() as conn:
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(SCHEMA)
    logger.info("Database reset complete")
