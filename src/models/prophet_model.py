"""Prophet wrapper: one additive model per (store, product) series."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data import schema as S
from src.models.base import BaseForecaster, _check_fitted

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

REGRESSORS = ["promotion", "holiday"]


class ProphetForecaster(BaseForecaster):
    """Fits per series, falls back to the global mean for unseen series.

    `max_series` caps training cost on wide panels; series beyond the cap are
    predicted from their own historical mean.
    """

    name = "prophet"
    needs_context = True

    def __init__(self, max_series: int = 60, **params: Any) -> None:
        super().__init__(max_series=max_series, **params)
        self.max_series = max_series
        self.models_: Dict[Any, Any] = {}
        self.series_mean_: Dict[Any, float] = {}
        self.regressors_: list[str] = []
        self.global_mean_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, context: Optional[pd.DataFrame] = None):
        from prophet import Prophet

        if context is None:
            raise ValueError("ProphetForecaster requires context with date/store/product")
        self.feature_names_ = list(X.columns)
        frame = context.copy()
        frame[S.TARGET] = np.asarray(y, dtype=float)
        self.regressors_ = [c for c in REGRESSORS if c in frame.columns]
        self.global_mean_ = float(frame[S.TARGET].mean())

        totals = frame.groupby([S.STORE, S.PRODUCT], observed=True)[S.TARGET].sum().sort_values(ascending=False)
        selected = set(totals.head(self.max_series).index)

        for key, group in frame.groupby([S.STORE, S.PRODUCT], observed=True):
            self.series_mean_[key] = float(group[S.TARGET].mean())
            if key not in selected or len(group) < 30:
                continue
            history = group.rename(columns={S.DATE: "ds", S.TARGET: "y"})[["ds", "y"] + self.regressors_]
            model = Prophet(
                seasonality_mode=self.params.get("seasonality_mode", "multiplicative"),
                weekly_seasonality=self.params.get("weekly_seasonality", True),
                yearly_seasonality=self.params.get("yearly_seasonality", True),
                daily_seasonality=False,
                uncertainty_samples=0,
            )
            for reg in self.regressors_:
                model.add_regressor(reg)
            model.fit(history)
            self.models_[key] = model

        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, context: Optional[pd.DataFrame] = None) -> np.ndarray:
        _check_fitted(self)
        if context is None:
            raise ValueError("ProphetForecaster requires context with date/store/product")
        out = np.full(len(context), self.global_mean_, dtype=float)
        positions = np.arange(len(context))
        for key, group in context.groupby([S.STORE, S.PRODUCT], observed=True):
            idx = positions[context.index.get_indexer(group.index)]
            model = self.models_.get(key)
            if model is None:
                out[idx] = self.series_mean_.get(key, self.global_mean_)
                continue
            future = group.rename(columns={S.DATE: "ds"})[["ds"] + self.regressors_].copy()
            for reg in self.regressors_:
                future[reg] = pd.to_numeric(future[reg], errors="coerce").fillna(0)
            out[idx] = model.predict(future)["yhat"].to_numpy(dtype=float)
        return np.clip(out, 0, None)

    def feature_importance(self):  # Prophet has no feature importances
        return None
