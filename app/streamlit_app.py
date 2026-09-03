"""Home page of the retail forecasting dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.components import state  # noqa: E402
from src.config import Config  # noqa: E402

state.set_page("Retail Sales Forecasting", "📈")
config: Config = state.get_config()

st.markdown(
    """
Forecast demand, explain the drivers and turn both into decisions.
Use the sidebar to move through the workflow: **Upload → Validation → EDA → Features →
Models → Forecast → Explainability → Insights → Scenario Analysis**.
"""
)

left, right = st.columns([2, 1])

with left:
    st.subheader("Status")
    clean = state.get("clean")
    bundle = state.load_saved_model()
    rows = [
        ("Dataset", state.get("source_name") or "not loaded"),
        ("Rows", f"{len(clean):,}" if clean is not None else "-"),
        ("Date range", f"{clean['date'].min().date()} → {clean['date'].max().date()}" if clean is not None else "-"),
        ("Trained model", bundle.model_name if bundle else "none"),
        ("Model RMSE (CV)", f"{bundle.metrics.get('rmse', float('nan')):,.2f}" if bundle else "-"),
    ]
    for label, value in rows:
        st.write(f"**{label}:** {value}")

with right:
    st.subheader("Quick start")
    if st.button("Load bundled sample dataset", use_container_width=True):
        with st.spinner("Loading sample..."):
            state.load_sample()
        st.success("Sample loaded. Continue on the Validation page.")
    st.caption(f"Schema preset: `{config.get('data.schema_preset')}`")

if clean is not None:
    from src.eda import analysis

    st.divider()
    st.subheader("Headline KPIs")
    kpis = analysis.kpi_summary(clean)
    columns = st.columns(4)
    columns[0].metric("Total sales", f"{kpis['total_sales']:,.0f}")
    columns[1].metric("Avg daily sales", f"{kpis['avg_daily_sales']:,.0f}")
    columns[2].metric("Stores × products", f"{kpis['n_stores']} × {kpis['n_products']}")
    columns[3].metric("28d growth", f"{kpis['mom_growth_pct']:+.1f}%")

st.divider()
st.caption(
    "Pipeline: upload → validation → cleaning → feature engineering → walk-forward training → "
    "explainability → forecast → insights. Run headless with `python scripts/run_pipeline.py`."
)
