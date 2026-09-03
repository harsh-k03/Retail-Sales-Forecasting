"""Deterministic cleaning steps applied after validation."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.data import schema as S
from src.logger import get_logger

logger = get_logger(__name__)


def clean(frame: pd.DataFrame, config: Optional[Config] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Return the cleaned frame plus a per-step record of rows affected."""
    config = config or Config.load()
    rules = config.get("cleaning", {}) or {}
    stats: Dict[str, int] = {}
    out = frame.copy()

    before = len(out)
    out = out.dropna(subset=[S.DATE])
    stats["dropped_invalid_dates"] = before - len(out)

    if rules.get("drop_duplicates", True):
        before = len(out)
        out = out.sort_values(S.KEY_COLUMNS).drop_duplicates(subset=S.KEY_COLUMNS, keep="last")
        stats["dropped_duplicates"] = before - len(out)

    out = _handle_missing_target(out, rules.get("fill_missing_target", "drop"), stats)

    if rules.get("clip_negative_sales", True):
        negatives = int((out[S.TARGET] < 0).sum())
        out[S.TARGET] = out[S.TARGET].clip(lower=0)
        stats["clipped_negatives"] = negatives

    out = _handle_outliers(out, rules, stats)
    out = _fill_exogenous(out, stats)

    out = out.sort_values(S.KEY_COLUMNS).reset_index(drop=True)
    logger.info("Cleaning stats: %s", stats)
    return out, stats


def _handle_missing_target(frame: pd.DataFrame, policy: str, stats: Dict[str, int]) -> pd.DataFrame:
    missing = int(frame[S.TARGET].isna().sum())
    stats["missing_target"] = missing
    if not missing:
        return frame
    if policy == "drop":
        return frame.dropna(subset=[S.TARGET])
    if policy == "zero":
        frame[S.TARGET] = frame[S.TARGET].fillna(0)
        return frame
    filled = (
        frame.sort_values(S.KEY_COLUMNS)
        .groupby([S.STORE, S.PRODUCT], observed=True)[S.TARGET]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    frame[S.TARGET] = filled.fillna(0)
    return frame


def _handle_outliers(frame: pd.DataFrame, rules: Dict, stats: Dict[str, int]) -> pd.DataFrame:
    method = rules.get("outlier_method", "none")
    if method == "none":
        stats["outliers"] = 0
        return frame
    factor = float(rules.get("outlier_factor", 3.0))
    action = rules.get("outlier_action", "clip")

    grouped = frame.groupby([S.STORE, S.PRODUCT], observed=True)[S.TARGET]
    if method == "iqr":
        q1 = grouped.transform(lambda s: s.quantile(0.25))
        q3 = grouped.transform(lambda s: s.quantile(0.75))
        iqr = (q3 - q1).replace(0, np.nan)
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
    else:  # zscore
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        lower, upper = mean - factor * std, mean + factor * std

    lower = lower.fillna(-np.inf)
    upper = upper.fillna(np.inf)
    mask = (frame[S.TARGET] < lower) | (frame[S.TARGET] > upper)
    stats["outliers"] = int(mask.sum())

    if action == "drop":
        return frame.loc[~mask]
    if action == "flag":
        frame["is_outlier"] = mask.astype("int8")
        return frame
    frame[S.TARGET] = frame[S.TARGET].clip(lower=lower, upper=upper)
    return frame


def _fill_exogenous(frame: pd.DataFrame, stats: Dict[str, int]) -> pd.DataFrame:
    """Forward/median fill optional drivers so features stay dense."""
    filled = 0
    for col in ("temperature", "inventory", "marketing_spend"):
        if col in frame.columns and frame[col].isna().any():
            filled += int(frame[col].isna().sum())
            by_series = frame.groupby([S.STORE, S.PRODUCT], observed=True)[col]
            frame[col] = by_series.transform(lambda s: s.ffill().bfill())
            frame[col] = frame[col].fillna(frame[col].median())
    for col in ("promotion", "holiday"):
        if col in frame.columns and frame[col].isna().any():
            filled += int(frame[col].isna().sum())
            frame[col] = frame[col].fillna(0).astype("int8")
    for col in ("region", "category"):
        if col in frame.columns and frame[col].isna().any():
            filled += int(frame[col].isna().sum())
            frame[col] = frame[col].fillna("unknown")
    stats["filled_exogenous"] = filled
    return frame


def reindex_full_calendar(frame: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Expand each (store, product) series onto a gap-free calendar."""
    pieces = []
    for (store, product), group in frame.groupby([S.STORE, S.PRODUCT], observed=True):
        idx = pd.date_range(group[S.DATE].min(), group[S.DATE].max(), freq=freq)
        g = group.set_index(S.DATE).reindex(idx)
        g[S.STORE], g[S.PRODUCT] = store, product
        g[S.TARGET] = g[S.TARGET].fillna(0)
        g.index.name = S.DATE
        pieces.append(g.reset_index())
    return pd.concat(pieces, ignore_index=True).sort_values(S.KEY_COLUMNS).reset_index(drop=True)
