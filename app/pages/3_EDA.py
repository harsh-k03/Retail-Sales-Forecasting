"""Exploratory analysis of the cleaned dataset."""
from __future__ import annotations

import streamlit as st

from app.components import charts, state
from src.eda import analysis

state.set_page("Exploratory Analysis", "🔍")
if not state.require_data():
    st.stop()

frame = state.get("clean")
stores = sorted(frame["store"].unique())
products = sorted(frame["product"].unique())

with st.sidebar:
    st.header("Filters")
    picked_stores = st.multiselect("Stores", stores, default=stores[: min(len(stores), 8)])
    picked_products = st.multiselect("Products", products, default=products)

view = frame[frame["store"].isin(picked_stores) & frame["product"].isin(picked_products)]
if view.empty:
    st.warning("No rows match the current filters.")
    st.stop()

kpis = analysis.kpi_summary(view)
cols = st.columns(5)
cols[0].metric("Total sales", f"{kpis['total_sales']:,.0f}")
cols[1].metric("Avg daily", f"{kpis['avg_daily_sales']:,.0f}")
cols[2].metric("Peak day", f"{kpis['peak_daily_sales']:,.0f}")
cols[3].metric("28d growth", f"{kpis['mom_growth_pct']:+.1f}%")
cols[4].metric("Zero-sales rows", f"{kpis['zero_sales_ratio']:.1%}")

st.subheader("Trend and moving averages")
st.plotly_chart(charts.line_trend(analysis.daily_totals(view)), use_container_width=True)

left, right = st.columns(2)
season = analysis.seasonality_table(view)
left.plotly_chart(charts.bar(season["dayofweek"], "dayofweek", "sales", "Average sales by weekday"), use_container_width=True)
right.plotly_chart(charts.bar(season["month"], "month", "sales", "Average sales by month"), use_container_width=True)

st.subheader("Store and product performance")
left, right = st.columns(2)
left.dataframe(analysis.store_performance(view), use_container_width=True, hide_index=True)
right.plotly_chart(
    charts.bar(analysis.product_performance(view, 15), "product", "total_sales", "Top products", horizontal=True),
    use_container_width=True,
)

st.subheader("Store × month heatmap")
st.plotly_chart(charts.heatmap(analysis.store_month_heatmap(view)), use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Promotion effect", "Anomalies", "Series stability"])
with tab1:
    promo = analysis.promotion_effect(view)
    st.dataframe(promo, use_container_width=True, hide_index=True) if promo is not None else st.info("No promotion column.")
with tab2:
    st.dataframe(analysis.detect_anomalies(view).head(50), use_container_width=True, hide_index=True)
with tab3:
    st.caption("Coefficient of variation per series - high values are intrinsically hard to forecast.")
    st.dataframe(analysis.series_stability(view).head(50), use_container_width=True, hide_index=True)
