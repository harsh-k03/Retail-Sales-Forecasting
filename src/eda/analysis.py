"""Exploratory analysis: tabular summaries used by both reports and dashboard."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data import schema as S


def kpi_summary(frame: pd.DataFrame) -> Dict[str, float]:
    daily = frame.groupby(S.DATE, observed=True)[S.TARGET].sum()
    last_28 = daily.tail(28).mean() if len(daily) >= 28 else daily.mean()
    prev_28 = daily.tail(56).head(28).mean() if len(daily) >= 56 else np.nan
    growth = (last_28 / prev_28 - 1) * 100 if prev_28 and not np.isnan(prev_28) else np.nan
    return {
        "total_sales": float(frame[S.TARGET].sum()),
        "avg_daily_sales": float(daily.mean()),
        "median_daily_sales": float(daily.median()),
        "peak_daily_sales": float(daily.max()),
        "n_stores": int(frame[S.STORE].nunique()),
        "n_products": int(frame[S.PRODUCT].nunique()),
        "n_days": int(frame[S.DATE].nunique()),
        "mom_growth_pct": float(growth) if growth == growth else 0.0,
        "zero_sales_ratio": float((frame[S.TARGET] == 0).mean()),
    }


def daily_totals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.groupby(S.DATE, observed=True)[S.TARGET].sum().reset_index()
    out["ma_7"] = out[S.TARGET].rolling(7, min_periods=1).mean()
    out["ma_28"] = out[S.TARGET].rolling(28, min_periods=1).mean()
    return out


def seasonality_table(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    work = frame.copy()
    work["dayofweek"] = work[S.DATE].dt.day_name()
    work["month"] = work[S.DATE].dt.month_name()
    order_dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = work.groupby("dayofweek", observed=True)[S.TARGET].mean().reindex(order_dow).reset_index()
    month = work.groupby(work[S.DATE].dt.month, observed=True)[S.TARGET].mean().reset_index()
    month.columns = ["month", S.TARGET]
    return {"dayofweek": dow, "month": month}


def store_performance(frame: pd.DataFrame) -> pd.DataFrame:
    out = (
        frame.groupby(S.STORE, observed=True)
        .agg(total_sales=(S.TARGET, "sum"), avg_sales=(S.TARGET, "mean"), days=(S.DATE, "nunique"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )
    out["share_pct"] = out["total_sales"] / out["total_sales"].sum() * 100
    return out


def product_performance(frame: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    out = (
        frame.groupby(S.PRODUCT, observed=True)
        .agg(total_sales=(S.TARGET, "sum"), avg_sales=(S.TARGET, "mean"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )
    out["share_pct"] = out["total_sales"] / out["total_sales"].sum() * 100
    return out.head(top_n)


def store_month_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = work[S.DATE].dt.to_period("M").astype(str)
    return work.pivot_table(index=S.STORE, columns="month", values=S.TARGET, aggfunc="sum", observed=True).fillna(0)


def promotion_effect(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "promotion" not in frame.columns:
        return None
    out = (
        frame.groupby("promotion", observed=True)[S.TARGET]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"promotion": "on_promotion"})
    )
    if len(out) == 2:
        base = out.loc[out["on_promotion"] == 0, "mean"].iloc[0]
        promo = out.loc[out["on_promotion"] == 1, "mean"].iloc[0]
        out["uplift_pct"] = [0.0, (promo / base - 1) * 100 if base else 0.0]
    return out


def holiday_effect(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "holiday" not in frame.columns:
        return None
    return frame.groupby("holiday", observed=True)[S.TARGET].agg(["mean", "count"]).reset_index()


def detect_anomalies(frame: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag days whose total sales deviate strongly from a 28-day rolling baseline."""
    daily = frame.groupby(S.DATE, observed=True)[S.TARGET].sum().reset_index()
    roll_mean = daily[S.TARGET].rolling(28, min_periods=7).mean()
    roll_std = daily[S.TARGET].rolling(28, min_periods=7).std().replace(0, np.nan)
    daily["z_score"] = (daily[S.TARGET] - roll_mean) / roll_std
    anomalies = daily.loc[daily["z_score"].abs() >= z_threshold].copy()
    anomalies["direction"] = np.where(anomalies["z_score"] > 0, "spike", "drop")
    return anomalies.sort_values("z_score", key=np.abs, ascending=False)


def series_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation per series - high values are hard to forecast."""
    grouped = frame.groupby([S.STORE, S.PRODUCT], observed=True)[S.TARGET]
    out = grouped.agg(mean="mean", std="std", zeros=lambda s: float((s == 0).mean())).reset_index()
    out["cv"] = out["std"] / out["mean"].replace(0, np.nan)
    return out.sort_values("cv", ascending=False)


def run_eda(frame: pd.DataFrame) -> Dict[str, object]:
    """Everything the EDA page and report need, in one call."""
    return {
        "kpis": kpi_summary(frame),
        "daily": daily_totals(frame),
        "seasonality": seasonality_table(frame),
        "stores": store_performance(frame),
        "products": product_performance(frame),
        "heatmap": store_month_heatmap(frame),
        "promotion": promotion_effect(frame),
        "holiday": holiday_effect(frame),
        "anomalies": detect_anomalies(frame),
        "stability": series_stability(frame),
    }
