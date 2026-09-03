"""Recursive multi-step forecasting for lag-based models."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.data import schema as S
from src.features.builder import FeatureBuilder
from src.features.calendar import future_frame
from src.logger import get_logger
from src.models.base import BaseForecaster

logger = get_logger(__name__)

CONTEXT_COLUMNS = [S.DATE, S.STORE, S.PRODUCT, "promotion", "holiday"]


def make_future_frame(
    history: pd.DataFrame,
    horizon: int,
    overrides: Optional[Dict[str, object]] = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Future skeleton with assumed drivers; `overrides` sets scenario values."""
    future = future_frame(history, horizon, freq=freq)
    for column, value in (overrides or {}).items():
        if column in future.columns or column in S.OPTIONAL_COLUMNS:
            future[column] = value
    return future


def forecast(
    model: BaseForecaster,
    history: pd.DataFrame,
    builder: FeatureBuilder,
    horizon: int = 30,
    overrides: Optional[Dict[str, object]] = None,
    window_days: int = 540,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """Predict `horizon` steps ahead, feeding each prediction back as a lag.

    `window_days` caps how much history is rebuilt per step; lag/rolling windows
    are far shorter than the default, so the trade-off is speed for an exact
    expanding-mean feature.
    """
    config = config or Config.load()
    freq = config.get("data.frequency", "D")
    future = make_future_frame(history, horizon, overrides, freq=freq)
    combined = pd.concat([history.copy(), future], ignore_index=True).sort_values(S.KEY_COLUMNS)
    combined = combined.reset_index(drop=True)

    step_dates = sorted(future[S.DATE].unique())
    out_rows: List[pd.DataFrame] = []

    for step, current in enumerate(step_dates, start=1):
        window_start = pd.Timestamp(current) - pd.Timedelta(days=window_days)
        window = combined[(combined[S.DATE] >= window_start) & (combined[S.DATE] <= current)]
        featured = builder.transform(window)
        mask = featured[S.DATE] == current
        block = featured.loc[mask]
        if block.empty:
            continue

        X = block[builder.spec.feature_names].fillna(0)
        context = block[[c for c in CONTEXT_COLUMNS if c in block.columns]]
        preds = np.clip(np.asarray(model.predict(X, context=context), dtype=float), 0, None)

        keys = block[[S.DATE, S.STORE, S.PRODUCT]].copy()
        keys["forecast"] = preds
        keys["step"] = step
        out_rows.append(keys)

        # feed predictions back so the next step sees them as lags
        idx = combined.index[combined[S.DATE] == current]
        lookup = keys.set_index([S.STORE, S.PRODUCT])["forecast"]
        target_keys = pd.MultiIndex.from_frame(combined.loc[idx, [S.STORE, S.PRODUCT]])
        combined.loc[idx, S.TARGET] = lookup.reindex(target_keys).to_numpy()

    if not out_rows:
        return pd.DataFrame(columns=[S.DATE, S.STORE, S.PRODUCT, "forecast", "step"])
    result = pd.concat(out_rows, ignore_index=True).sort_values([S.DATE, S.STORE, S.PRODUCT])
    logger.info("Forecast produced %d rows over %d steps", len(result), len(step_dates))
    return result.reset_index(drop=True)


def forecast_horizons(
    model: BaseForecaster,
    history: pd.DataFrame,
    builder: FeatureBuilder,
    horizons: Optional[List[int]] = None,
    config: Optional[Config] = None,
    **kwargs,
) -> Dict[int, pd.DataFrame]:
    """Run the longest horizon once and slice the shorter ones out of it."""
    config = config or Config.load()
    horizons = sorted(horizons or config.get("forecast.horizons", [7, 30, 90]))
    full = forecast(model, history, builder, horizon=max(horizons), config=config, **kwargs)
    return {h: full[full["step"] <= h].reset_index(drop=True) for h in horizons}


def aggregate_forecast(forecast_frame: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    if by is None:
        return forecast_frame.groupby(S.DATE, observed=True)["forecast"].sum().reset_index()
    return forecast_frame.groupby([S.DATE, by], observed=True)["forecast"].sum().reset_index()


def forecast_summary(forecast_frame: pd.DataFrame, history: pd.DataFrame) -> Dict[str, float]:
    """Headline numbers comparing the forecast window to the recent past."""
    horizon = int(forecast_frame["step"].max()) if len(forecast_frame) else 0
    total = float(forecast_frame["forecast"].sum())
    recent = history[history[S.DATE] > history[S.DATE].max() - pd.Timedelta(days=horizon)]
    recent_total = float(recent[S.TARGET].sum())
    return {
        "horizon_days": horizon,
        "total_forecast": total,
        "avg_daily_forecast": total / max(horizon, 1),
        "recent_actual_total": recent_total,
        "change_vs_recent_pct": (total / recent_total - 1) * 100 if recent_total else 0.0,
    }
