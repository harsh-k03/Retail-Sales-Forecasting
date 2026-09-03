"""Baselines every learned model must beat."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data import schema as S
from src.models.base import BaseForecaster, _check_fitted


class SeasonalNaiveForecaster(BaseForecaster):
    """Predicts the value from `season_length` periods ago (weekly by default)."""

    name = "naive"

    def __init__(self, season_length: int = 7, **params) -> None:
        super().__init__(season_length=season_length, **params)
        self.season_length = season_length
        self.fallback_ = 0.0
        self.lag_column_: Optional[str] = None

    def _pick_column(self, X: pd.DataFrame) -> Optional[str]:
        preferred = f"{S.TARGET}_lag_{self.season_length}"
        for candidate in (preferred, f"{S.TARGET}_lag_1", f"{S.TARGET}_roll7_mean"):
            if candidate in X.columns:
                return candidate
        return None

    def fit(self, X: pd.DataFrame, y: pd.Series, context: Optional[pd.DataFrame] = None) -> "SeasonalNaiveForecaster":
        self.feature_names_ = list(X.columns)
        self.lag_column_ = self._pick_column(X)
        self.fallback_ = float(np.nanmean(y)) if len(y) else 0.0
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, context: Optional[pd.DataFrame] = None) -> np.ndarray:
        _check_fitted(self)
        if self.lag_column_ and self.lag_column_ in X.columns:
            values = pd.to_numeric(X[self.lag_column_], errors="coerce").to_numpy(dtype=float)
        else:
            values = np.full(len(X), np.nan)
        return np.nan_to_num(values, nan=self.fallback_)


class MovingAverageForecaster(BaseForecaster):
    """Rolling-mean baseline, useful when series are noisy and weakly seasonal."""

    name = "moving_average"

    def __init__(self, window: int = 28, **params) -> None:
        super().__init__(window=window, **params)
        self.window = window
        self.fallback_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, context: Optional[pd.DataFrame] = None) -> "MovingAverageForecaster":
        self.feature_names_ = list(X.columns)
        self.fallback_ = float(np.nanmean(y)) if len(y) else 0.0
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, context: Optional[pd.DataFrame] = None) -> np.ndarray:
        _check_fitted(self)
        col = f"{S.TARGET}_roll{self.window}_mean"
        if col not in X.columns:
            col = next((c for c in X.columns if c.startswith(f"{S.TARGET}_roll")), None)
        if col is None:
            return np.full(len(X), self.fallback_)
        return np.nan_to_num(X[col].to_numpy(dtype=float), nan=self.fallback_)
