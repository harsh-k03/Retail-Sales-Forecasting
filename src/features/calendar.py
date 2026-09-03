"""Calendar, seasonality and event features."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from src.data import schema as S

FOURIER_EPOCH = pd.Timestamp("2000-01-01")


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    d = out[S.DATE].dt
    out["year"] = d.year.astype("int16")
    out["month"] = d.month.astype("int8")
    out["day"] = d.day.astype("int8")
    out["dayofweek"] = d.dayofweek.astype("int8")
    out["dayofyear"] = d.dayofyear.astype("int16")
    out["weekofyear"] = d.isocalendar().week.astype("int16").to_numpy()
    out["quarter"] = d.quarter.astype("int8")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    out["is_month_start"] = d.is_month_start.astype("int8")
    out["is_month_end"] = d.is_month_end.astype("int8")
    out["is_quarter_end"] = d.is_quarter_end.astype("int8")
    out["days_in_month"] = d.days_in_month.astype("int8")
    # payday effect: many retail markets spike around the 15th and month end
    out["is_payday_window"] = ((out["day"].isin([14, 15, 16])) | (out["is_month_end"] == 1)).astype("int8")
    return out


def add_fourier_terms(
    frame: pd.DataFrame,
    periods: Sequence[float] = (7, 365.25),
    order: int = 3,
) -> pd.DataFrame:
    """Smooth seasonal basis; cheaper than one-hot day/week dummies."""
    out = frame.copy()
    # fixed epoch keeps the phase stable when only a tail window is rebuilt
    elapsed = (out[S.DATE] - FOURIER_EPOCH).dt.days.to_numpy()
    for period in periods:
        for k in range(1, order + 1):
            angle = 2 * np.pi * k * elapsed / period
            tag = f"{str(period).replace('.', '_')}_{k}"
            out[f"sin_{tag}"] = np.sin(angle)
            out[f"cos_{tag}"] = np.cos(angle)
    return out


def add_event_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Distance-to-event features around holidays and promotions."""
    out = frame.copy()
    for col in ("holiday", "promotion"):
        if col not in out.columns:
            continue
        grouped = out.groupby([S.STORE, S.PRODUCT], observed=True)[col]
        out[f"{col}_lead_1"] = grouped.shift(-1).fillna(0).astype("int8")
        out[f"{col}_lag_1"] = grouped.shift(1).fillna(0).astype("int8")
        out[f"{col}_roll_7"] = (
            grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
        )
    return out


def calendar_feature_names(frame: pd.DataFrame) -> List[str]:
    base = [
        "year", "month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter",
        "is_weekend", "is_month_start", "is_month_end", "is_quarter_end",
        "days_in_month", "is_payday_window",
    ]
    return [c for c in base if c in frame.columns]


def future_frame(history: pd.DataFrame, horizon: int, freq: str = "D") -> pd.DataFrame:
    """Empty future rows for every (store, product) series, ready for recursion."""
    pieces = []
    for (store, product), group in history.groupby([S.STORE, S.PRODUCT], observed=True):
        last = group[S.DATE].max()
        dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq=freq)
        block = pd.DataFrame({S.DATE: dates, S.STORE: store, S.PRODUCT: product})
        tail = group.sort_values(S.DATE).iloc[-1]
        for col in ("region", "category"):
            if col in group.columns:
                block[col] = tail[col]
        for col in ("promotion", "holiday"):
            if col in group.columns:
                block[col] = 0
        for col in ("temperature", "inventory", "marketing_spend"):
            if col in group.columns:
                block[col] = group[col].tail(28).mean()
        block[S.TARGET] = np.nan
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)
