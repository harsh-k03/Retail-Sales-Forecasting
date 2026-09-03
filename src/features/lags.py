"""Lag and rolling-window features. All windows are shifted to avoid leakage."""
from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from src.data import schema as S

GROUP_KEYS = [S.STORE, S.PRODUCT]


def add_lag_features(frame: pd.DataFrame, lags: Sequence[int], target: str = S.TARGET) -> pd.DataFrame:
    out = frame.sort_values(S.KEY_COLUMNS).copy()
    grouped = out.groupby(GROUP_KEYS, observed=True)[target]
    for lag in lags:
        out[f"{target}_lag_{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(
    frame: pd.DataFrame,
    windows: Sequence[int],
    stats: Sequence[str] = ("mean", "std"),
    target: str = S.TARGET,
) -> pd.DataFrame:
    """Rolling stats over the shifted series, so row t never sees y_t."""
    out = frame.sort_values(S.KEY_COLUMNS).copy()
    shifted = out.groupby(GROUP_KEYS, observed=True)[target].shift(1)
    out["_shifted"] = shifted
    grouped = out.groupby(GROUP_KEYS, observed=True)["_shifted"]
    for window in windows:
        min_periods = max(2, window // 4)
        for stat in stats:
            out[f"{target}_roll{window}_{stat}"] = grouped.transform(
                lambda s, w=window, st=stat, mp=min_periods: getattr(s.rolling(w, min_periods=mp), st)()
            )
    out = out.drop(columns=["_shifted"])
    return out


def add_expanding_features(frame: pd.DataFrame, target: str = S.TARGET) -> pd.DataFrame:
    out = frame.sort_values(S.KEY_COLUMNS).copy()
    grouped = out.groupby(GROUP_KEYS, observed=True)[target]
    out[f"{target}_expanding_mean"] = grouped.transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    return out


def add_diff_features(frame: pd.DataFrame, lags: Sequence[int] = (1, 7), target: str = S.TARGET) -> pd.DataFrame:
    out = frame.copy()
    for lag in lags:
        col = f"{target}_lag_{lag}"
        base = f"{target}_lag_{lag + 1}" if f"{target}_lag_{lag + 1}" in out.columns else f"{target}_lag_{lag}"
        if col in out.columns and base in out.columns and col != base:
            out[f"{target}_diff_{lag}"] = out[col] - out[base]
    if f"{target}_lag_1" in out.columns and f"{target}_lag_7" in out.columns:
        out[f"{target}_momentum_7"] = out[f"{target}_lag_1"] - out[f"{target}_lag_7"]
    return out


def lag_feature_names(frame: pd.DataFrame, target: str = S.TARGET) -> List[str]:
    prefix = (f"{target}_lag_", f"{target}_roll", f"{target}_expanding", f"{target}_diff_", f"{target}_momentum")
    return [c for c in frame.columns if c.startswith(prefix)]


def max_lag(lags: Sequence[int], windows: Sequence[int]) -> int:
    return max(list(lags) + [w + 1 for w in windows] + [1])
