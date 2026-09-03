"""Common forecaster interface so every model is swappable."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Minimal contract: fit on a feature matrix, predict on the same columns.

    `context` carries date/store/product for models (Prophet) that need the
    raw series rather than the engineered matrix.
    """

    name: str = "base"
    supports_shap: bool = False
    needs_context: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = params
        self.model: Any = None
        self.feature_names_: list[str] = []
        self.fitted_: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, context: Optional[pd.DataFrame] = None) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, context: Optional[pd.DataFrame] = None) -> np.ndarray:
        ...

    def feature_importance(self) -> Optional[pd.DataFrame]:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None or not self.feature_names_:
            return None
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": np.asarray(importances, dtype=float)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_params(self) -> Dict[str, Any]:
        return dict(self.params)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self.fitted_})"


def _check_fitted(model: BaseForecaster) -> None:
    if not model.fitted_:
        raise RuntimeError(f"{model.name} is not fitted yet")
