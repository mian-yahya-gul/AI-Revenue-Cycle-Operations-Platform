"""Shared Streamlit helpers: DB bootstrap, status badges, formatting."""

from __future__ import annotations

import streamlit as st

from database.db import init_db
from models.schemas import ClaimStatus, RiskLevel

STATUS_COLORS = {
    ClaimStatus.RECEIVED: "#64748b",
    ClaimStatus.IN_REVIEW: "#2563eb",
    ClaimStatus.READY_FOR_SUBMISSION: "#16a34a",
    ClaimStatus.NEEDS_HUMAN_REVIEW: "#d97706",
    ClaimStatus.SUBMITTED: "#0891b2",
    ClaimStatus.DENIED: "#dc2626",
    ClaimStatus.APPEALING: "#7c3aed",
    ClaimStatus.READY_FOR_RESUBMISSION: "#0d9488",
    ClaimStatus.RESUBMITTED: "#0891b2",
    ClaimStatus.APPROVED: "#16a34a",
    ClaimStatus.CLOSED: "#475569",
}

RISK_COLORS = {
    RiskLevel.LOW: "#16a34a",
    RiskLevel.MEDIUM: "#d97706",
    RiskLevel.HIGH: "#dc2626",
}


def bootstrap() -> None:
    """Ensure the DB schema exists. Safe to call on every page load."""
    init_db()


def status_badge(status: ClaimStatus) -> str:
    color = STATUS_COLORS.get(status, "#64748b")
    label = status.value.replace("_", " ").title()
    return (
        f'<span style="background-color:{color}22;color:{color};'
        f'padding:3px 10px;border-radius:12px;font-size:0.82rem;'
        f'font-weight:600;border:1px solid {color}55;">{label}</span>'
    )


def risk_badge(risk: RiskLevel) -> str:
    color = RISK_COLORS.get(risk, "#64748b")
    label = risk.value.upper()
    return (
        f'<span style="background-color:{color}22;color:{color};'
        f'padding:3px 10px;border-radius:12px;font-size:0.82rem;'
        f'font-weight:700;border:1px solid {color}55;">{label}</span>'
    )


def render_badge_html(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()
