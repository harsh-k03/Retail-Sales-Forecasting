"""What-if simulator over the forecast window."""
from __future__ import annotations

import streamlit as st

from app.components import charts, state
from src.insights.scenario import Scenario, ScenarioSimulator

state.set_page("Scenario Analysis", "🎛️")
if not (state.require_data() and state.require_model()):
    st.stop()

config = state.get_config()
bundle = state.get("bundle")
history = state.get("clean")
simulator = ScenarioSimulator(bundle.model, history, bundle.builder, config)

with st.sidebar:
    st.header("Levers")
    horizon = st.select_slider("Horizon (days)", options=config.get("forecast.horizons", [7, 30, 90]), value=30)
    promotion = st.select_slider("Promotion", options=["baseline", 0, 1], value="baseline")
    holiday = st.select_slider("Holiday flag", options=["baseline", 0, 1], value="baseline")
    overrides = {}
    if "temperature" in history.columns:
        temp = st.slider("Temperature (°C)", -10.0, 45.0, float(history["temperature"].tail(90).mean()), 0.5)
        if st.checkbox("Apply temperature override"):
            overrides["temperature"] = temp
    if "marketing_spend" in history.columns:
        spend = st.number_input("Marketing spend", value=float(history["marketing_spend"].tail(90).mean()))
        if st.checkbox("Apply marketing spend override"):
            overrides["marketing_spend"] = spend
    if "inventory" in history.columns:
        inv = st.number_input("Inventory / transactions", value=float(history["inventory"].tail(90).mean()))
        if st.checkbox("Apply inventory override"):
            overrides["inventory"] = inv
    run = st.button("Simulate", type="primary", use_container_width=True)

if promotion != "baseline":
    overrides["promotion"] = int(promotion)
if holiday != "baseline":
    overrides["holiday"] = int(holiday)

if run:
    if not overrides:
        st.warning("Set at least one lever away from baseline.")
        st.stop()
    scenario = Scenario("scenario", overrides, horizon)
    with st.spinner("Running baseline and scenario forecasts..."):
        result = simulator.compare(scenario)
        daily = simulator.daily_comparison(scenario)

    cols = st.columns(4)
    cols[0].metric("Baseline total", f"{result['baseline_total']:,.0f}")
    cols[1].metric("Scenario total", f"{result['scenario_total']:,.0f}")
    cols[2].metric("Delta", f"{result['delta']:+,.0f}")
    cols[3].metric("Delta %", f"{result['delta_pct']:+.2f}%")

    st.plotly_chart(charts.scenario_chart(daily, scenario.name), use_container_width=True)
    st.dataframe(daily, use_container_width=True, hide_index=True)
    st.caption(f"Levers applied: {scenario.describe()}")
else:
    st.info("Adjust the levers in the sidebar and press **Simulate**. "
            "Each run re-forecasts recursively with the modified drivers.")

st.divider()
with st.expander("Sensitivity sweep"):
    lever = st.selectbox("Lever", ["promotion", "holiday"])
    if st.button("Run sweep"):
        with st.spinner("Sweeping..."):
            sweep = simulator.sweep(lever, [0, 1], horizon=horizon)
        st.dataframe(sweep, use_container_width=True, hide_index=True)
