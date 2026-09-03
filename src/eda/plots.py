"""Matplotlib figures for the static report. Dashboard uses Plotly separately."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import schema as S
from src.eda import analysis

sns.set_theme(style="whitegrid", palette="deep")
FIGSIZE = (11, 4.5)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trend(frame, out_dir: Path) -> Path:
    daily = analysis.daily_totals(frame)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(daily[S.DATE], daily[S.TARGET], lw=0.7, alpha=0.45, label="daily")
    ax.plot(daily[S.DATE], daily["ma_7"], lw=1.6, label="7-day MA")
    ax.plot(daily[S.DATE], daily["ma_28"], lw=1.8, label="28-day MA")
    ax.set_title("Total daily sales with moving averages")
    ax.set_xlabel("")
    ax.set_ylabel("sales")
    ax.legend()
    return _save(fig, out_dir / "trend.png")


def plot_seasonality(frame, out_dir: Path) -> Path:
    tables = analysis.seasonality_table(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=tables["dayofweek"], x="dayofweek", y=S.TARGET, ax=axes[0])
    axes[0].set_title("Average sales by day of week")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_xlabel("")
    sns.lineplot(data=tables["month"], x="month", y=S.TARGET, marker="o", ax=axes[1])
    axes[1].set_title("Average sales by month")
    return _save(fig, out_dir / "seasonality.png")


def plot_distribution(frame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(frame[S.TARGET], bins=60, ax=axes[0])
    axes[0].set_title("Sales distribution")
    sns.boxplot(data=frame, x=S.STORE, y=S.TARGET, ax=axes[1])
    axes[1].set_title("Sales by store")
    return _save(fig, out_dir / "distribution.png")


def plot_heatmap(frame, out_dir: Path) -> Path:
    pivot = analysis.store_month_heatmap(frame)
    fig, ax = plt.subplots(figsize=(12, max(3, 0.45 * len(pivot))))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax, cbar_kws={"label": "sales"})
    ax.set_title("Sales by store and month")
    return _save(fig, out_dir / "store_month_heatmap.png")


def plot_product_performance(frame, out_dir: Path) -> Path:
    products = analysis.product_performance(frame, top_n=15)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(products))))
    sns.barplot(data=products, y=S.PRODUCT, x="total_sales", ax=ax)
    ax.set_title("Top products by total sales")
    return _save(fig, out_dir / "product_performance.png")


def plot_forecast(history, forecast, out_dir: Path, tail_days: int = 120) -> Path:
    hist = history.groupby(S.DATE, observed=True)[S.TARGET].sum().tail(tail_days)
    fut = forecast.groupby(S.DATE, observed=True)["forecast"].sum()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hist.index, hist.to_numpy(), label="history")
    ax.plot(fut.index, fut.to_numpy(), label="forecast", ls="--")
    ax.axvline(hist.index.max(), color="grey", lw=1, ls=":")
    ax.set_title("Aggregate forecast")
    ax.legend()
    return _save(fig, out_dir / "forecast.png")


def plot_model_comparison(metrics_frame, out_dir: Path, metric: str = "rmse") -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    data = metrics_frame.sort_values(metric)
    sns.barplot(data=data, x=metric, y="model", ax=ax)
    ax.set_title(f"Model comparison ({metric.upper()}, lower is better)")
    return _save(fig, out_dir / "model_comparison.png")


def generate_all(frame, out_dir: Path) -> Dict[str, str]:
    out_dir = Path(out_dir)
    figures: List[Path] = [
        plot_trend(frame, out_dir),
        plot_seasonality(frame, out_dir),
        plot_distribution(frame, out_dir),
        plot_heatmap(frame, out_dir),
        plot_product_performance(frame, out_dir),
    ]
    return {p.stem: str(p) for p in figures}
