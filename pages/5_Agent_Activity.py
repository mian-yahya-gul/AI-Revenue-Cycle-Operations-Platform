"""Agent Activity page — observability dashboard over every agent
invocation and workflow event across the platform."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import repository
from utils.ui_helpers import bootstrap, page_header

st.set_page_config(page_title="Agent Activity", page_icon="🤖", layout="wide")
bootstrap()
page_header("Agent Activity", "Observability across every agent invocation and workflow event")

logs = repository.list_agent_logs(limit=500)

if not logs:
    st.info("No agent activity recorded yet. Submit a claim or seed sample data.")
    st.stop()

df = pd.DataFrame(logs)

col1, col2, col3 = st.columns(3)
col1.metric("Total Agent Invocations", len(df))
col2.metric("Distinct Claims Touched", df["claim_id"].nunique())
col3.metric("Avg. Duration (ms)", f"{df['duration_ms'].mean():.1f}")

st.markdown("#### Invocations by Agent")
agent_counts = df["agent_name"].value_counts()
st.bar_chart(agent_counts)

st.markdown("#### Average Duration by Agent (ms)")
avg_duration = df.groupby("agent_name")["duration_ms"].mean().sort_values(ascending=False)
st.bar_chart(avg_duration)

st.markdown("#### Recent Activity Log")
agent_filter = st.multiselect("Filter by agent", options=sorted(df["agent_name"].unique()))
filtered = df[df["agent_name"].isin(agent_filter)] if agent_filter else df

st.dataframe(
    filtered[["created_at", "claim_id", "agent_name", "action", "duration_ms", "output_summary"]]
    .rename(columns={
        "created_at": "Timestamp", "claim_id": "Claim ID", "agent_name": "Agent",
        "action": "Action", "duration_ms": "Duration (ms)", "output_summary": "Output",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("#### Event Trail (Event-Driven Workflow)")
events = repository.list_events(limit=500)
if events:
    edf = pd.DataFrame(events)
    st.dataframe(
        edf[["created_at", "claim_id", "event_type"]]
        .rename(columns={"created_at": "Timestamp", "claim_id": "Claim ID", "event_type": "Event"}),
        use_container_width=True,
        hide_index=True,
    )
