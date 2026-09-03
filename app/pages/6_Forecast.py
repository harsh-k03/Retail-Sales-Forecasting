"""Generate and download multi-horizon forecasts."""
from __future__ import annotations

import streamlit as st

from app.components import charts, state
from src.forecast.recursive import forecast as run_forecast
from src.forecast.recursive import forecast_summary

state.set_page("Forecast", "🔮")
if not (state.require_data() and state.require_model()):
    st.stop()

config = state.get_config()
bundle = state.get("bundle")
history = state.get("clean")

with st.sidebar:
    st.header("Forecast settings")
    horizons = config.get("forecast.horizons", [7, 30, 90])
    horizon = st.select_slider("Horizon (days)", options=horizons, value=config.get("forecast.default_horizon", 30))
    stores = ["all"] + sorted(history["store"].unique())
    store = st.selectbox("Store", stores)
    products = ["all"] + sorted(history["product"].unique())
    product = st.selectbox("Product", products)
    go = st.button("Run forecast", type="primary", use_container_width=True)

view = history
if store != "all":
    view = view[view["store"] == store]
if product != "all":
    view = view[view["product"] == product]

if go:
    with st.spinner(f"Forecasting {horizon} days with {bundle.model_name}..."):
        state.put("forecast", run_forecast(bundle.model, view, bundle.builder, horizon=horizon, config=config))

frame = state.get("forecast")
if frame is None:
    st.info("Choose a horizon in the sidebar and run the forecast.")
    st.stop()

summary = forecast_summary(frame, view)
cols = st.columns(4)
cols[0].metric("Horizon", f"{summary['horizon_days']} days")
cols[1].metric("Total forecast", f"{summary['total_forecast']:,.0f}")
cols[2].metric("Avg per day", f"{summary['avg_daily_forecast']:,.0f}")
cols[3].metric("vs recent actual", f"{summary['change_vs_recent_pct']:+.1f}%")

st.plotly_chart(charts.forecast_chart(view, frame), use_container_width=True)

left, right = st.columns(2)
left.subheader("By store")
left.dataframe(frame.groupby("store", observed=True)["forecast"].sum().reset_index(), use_container_width=True, hide_index=True)
right.subheader("By product")
right.dataframe(frame.groupby("product", observed=True)["forecast"].sum().reset_index(), use_container_width=True, hide_index=True)

st.subheader("Forecast rows")
st.dataframe(frame.head(500), use_container_width=True, hide_index=True)
st.download_button("Download CSV", frame.to_csv(index=False), file_name=f"forecast_{summary['horizon_days']}d.csv",
                   mime="text/csv")
