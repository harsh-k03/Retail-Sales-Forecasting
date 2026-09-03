"""Plotly chart builders used across dashboard pages."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data import schema as S

TEMPLATE = "plotly_white"


def line_trend(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=daily[S.DATE], y=daily[S.TARGET], name="daily", line=dict(width=1), opacity=0.45)
    fig.add_scatter(x=daily[S.DATE], y=daily["ma_7"], name="7-day MA", line=dict(width=2))
    fig.add_scatter(x=daily[S.DATE], y=daily["ma_28"], name="28-day MA", line=dict(width=2.5))
    fig.update_layout(template=TEMPLATE, height=380, margin=dict(t=30, b=10), yaxis_title="sales")
    return fig


def bar(frame: pd.DataFrame, x: str, y: str, title: str = "", horizontal: bool = False) -> go.Figure:
    fig = px.bar(frame, x=y if horizontal else x, y=x if horizontal else y,
                 orientation="h" if horizontal else "v", title=title, template=TEMPLATE)
    fig.update_layout(height=380, margin=dict(t=40, b=10))
    return fig


def heatmap(pivot: pd.DataFrame, title: str = "") -> go.Figure:
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlGnBu", title=title, template=TEMPLATE)
    fig.update_layout(height=420, margin=dict(t=40, b=10))
    return fig


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, tail_days: int = 120) -> go.Figure:
    hist = history.groupby(S.DATE, observed=True)[S.TARGET].sum().tail(tail_days)
    fut = forecast.groupby(S.DATE, observed=True)["forecast"].sum()
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.to_numpy(), name="history", line=dict(width=2))
    fig.add_scatter(x=fut.index, y=fut.to_numpy(), name="forecast", line=dict(width=2, dash="dash"))
    fig.add_vline(x=hist.index.max(), line_width=1, line_dash="dot", line_color="grey")
    fig.update_layout(template=TEMPLATE, height=420, margin=dict(t=30, b=10), yaxis_title="sales")
    return fig


def actual_vs_predicted(predictions: pd.DataFrame) -> go.Figure:
    daily = predictions.groupby(S.DATE, observed=True)[["actual", "prediction"]].sum().reset_index()
    fig = go.Figure()
    fig.add_scatter(x=daily[S.DATE], y=daily["actual"], name="actual", line=dict(width=2))
    fig.add_scatter(x=daily[S.DATE], y=daily["prediction"], name="predicted", line=dict(width=2, dash="dash"))
    fig.update_layout(template=TEMPLATE, height=380, margin=dict(t=30, b=10))
    return fig


def residual_plot(predictions: pd.DataFrame) -> go.Figure:
    residuals = predictions["actual"] - predictions["prediction"]
    fig = px.histogram(residuals, nbins=60, title="Residual distribution", template=TEMPLATE)
    fig.update_layout(height=350, showlegend=False, margin=dict(t=40, b=10), xaxis_title="actual - predicted")
    return fig


def scenario_chart(comparison: pd.DataFrame, scenario_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=comparison[S.DATE], y=comparison["baseline"], name="baseline", line=dict(width=2))
    fig.add_scatter(x=comparison[S.DATE], y=comparison[scenario_name], name=scenario_name,
                    line=dict(width=2, dash="dash"))
    fig.update_layout(template=TEMPLATE, height=400, margin=dict(t=30, b=10), yaxis_title="forecast sales")
    return fig


def kpi_row(container, kpis: dict, keys: Optional[list] = None) -> None:
    keys = keys or list(kpis)[:4]
    columns = container.columns(len(keys))
    for column, key in zip(columns, keys):
        value = kpis.get(key, 0)
        label = key.replace("_", " ").title()
        column.metric(label, f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}")
