"""Validation report and cleaning summary."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import state

state.set_page("Validation", "✅")

report = state.get("validation")
if report is None:
    st.warning("Load a dataset on the Upload page first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{report.n_rows:,}")
col2.metric("Columns", report.n_columns)
col3.metric("Errors", len(report.errors))
col4.metric("Warnings", len(report.warnings))

st.subheader("Rule results")
issues = report.to_frame()
if issues.empty:
    st.success("Every validation rule passed.")
else:
    st.dataframe(issues, use_container_width=True)

st.subheader("Dataset summary")
summary = dict(report.summary)
missing = summary.pop("missing_by_column", {})
target = summary.pop("target", {})
st.json(summary, expanded=False)

left, right = st.columns(2)
with left:
    st.caption("Missing values by column")
    st.dataframe(
        pd.DataFrame(missing.items(), columns=["column", "missing"]).sort_values("missing", ascending=False),
        use_container_width=True, hide_index=True,
    )
with right:
    st.caption("Target statistics")
    st.dataframe(pd.DataFrame(target.items(), columns=["statistic", "value"]), use_container_width=True, hide_index=True)

stats = state.get("cleaning_stats") or {}
if stats:
    st.subheader("Cleaning applied")
    st.dataframe(
        pd.DataFrame(stats.items(), columns=["step", "rows affected"]),
        use_container_width=True, hide_index=True,
    )
