"""Gradient-boosted and bagged tree forecasters."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster, _check_fitted


class _SklearnStyleForecaster(BaseForecaster):
    """Shared fit/predict for any estimator exposing sklearn's API."""

    supports_shap = True

    def _build(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series, context: Optional[pd.DataFrame] = None):
        self.feature_names_ = list(X.columns)
        self.model = self._build()
        self.model.fit(X.to_numpy(dtype=float), np.asarray(y, dtype=float))
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, context: Optional[pd.DataFrame] = None) -> np.ndarray:
        _check_fitted(self)
        matrix = X[self.feature_names_].to_numpy(dtype=float)
        return np.asarray(self.model.predict(matrix), dtype=float)


class RandomForestForecaster(_SklearnStyleForecaster):
    name = "random_forest"

    def _build(self):
        from sklearn.ensemble import RandomForestRegressor

        params = {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 2, "n_jobs": -1}
        params.update(self.params)
        return RandomForestRegressor(random_state=params.pop("random_state", 42), **params)


class XGBoostForecaster(_SklearnStyleForecaster):
    name = "xgboost"

    def _build(self):
        from xgboost import XGBRegressor

        params = {
            "n_estimators": 600, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.9, "colsample_bytree": 0.9, "tree_method": "hist", "n_jobs": -1,
        }
        params.update(self.params)
        return XGBRegressor(random_state=params.pop("random_state", 42), **params)


class LightGBMForecaster(_SklearnStyleForecaster):
    name = "lightgbm"

    def _build(self):
        from lightgbm import LGBMRegressor

        params = {
            "n_estimators": 700, "learning_rate": 0.05, "num_leaves": 63,
            "subsample": 0.9, "colsample_bytree": 0.9, "verbose": -1, "n_jobs": -1,
        }
        params.update(self.params)
        return LGBMRegressor(random_state=params.pop("random_state", 42), **params)


class CatBoostForecaster(_SklearnStyleForecaster):
    name = "catboost"

    def _build(self):
        from catboost import CatBoostRegressor

        params = {"iterations": 600, "learning_rate": 0.05, "depth": 6, "verbose": 0, "allow_writing_files": False}
        params.update(self.params)
        return CatBoostRegressor(random_seed=params.pop("random_state", 42), **params)
