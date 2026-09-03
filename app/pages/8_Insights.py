"""Ranked, actionable business recommendations."""
from __future__ import annotations

import streamlit as st

from app.components import state
from src.insights.engine import InsightEngine

state.set_page("Business Insights", "💡")
if not state.require_data():
    st.stop()

history = state.get("clean")
forecast = state.get("forecast")
explanation = state.get("explanation")
bundle = state.get("bundle")

importance = None
if explanation is not None:
    importance = explanation.global_importance()
elif bundle is not None:
    importance = bundle.model.feature_importance()

if forecast is None:
    st.info("Run a forecast first to unlock demand and inventory recommendations.")

table = InsightEngine(history, forecast, importance).to_frame()
if table.empty:
    st.warning("No insights generated for this dataset.")
    st.stop()

priority = st.multiselect("Priority", ["high", "medium", "low"], default=["high", "medium", "low"])
view = table[table["priority"].isin(priority)]

colors = {"high": "🔴", "medium": "🟠", "low": "🟢"}
for _, row in view.iterrows():
    with st.container(border=True):
        st.markdown(f"{colors.get(row['priority'], '')} **{row['title']}**  ·  `{row['category']}`")
        st.write(row["detail"])
        st.markdown(f"**Recommended action:** {row['action']}")

st.divider()
st.download_button("Download insights CSV", view.to_csv(index=False), "insights.csv", "text/csv")
