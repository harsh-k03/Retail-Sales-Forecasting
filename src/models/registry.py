"""Name -> forecaster factory used by training, tuning and the API."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.config import Config
from src.models.base import BaseForecaster
from src.models.naive import MovingAverageForecaster, SeasonalNaiveForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.tree import (
    CatBoostForecaster,
    LightGBMForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

_REGISTRY: Dict[str, Callable[..., BaseForecaster]] = {
    "naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "random_forest": RandomForestForecaster,
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
    "catboost": CatBoostForecaster,
    "prophet": ProphetForecaster,
}


def available_models() -> List[str]:
    return list(_REGISTRY)


def is_available(name: str) -> bool:
    """True when the model's optional dependency is importable."""
    optional = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost", "prophet": "prophet"}
    module = optional.get(name)
    if module is None:
        return name in _REGISTRY
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def create_model(name: str, config: Optional[Config] = None, **overrides: Any) -> BaseForecaster:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {available_models()}")
    config = config or Config.load()
    params = config.defaults_for(name)
    params.update(overrides)
    return _REGISTRY[name](**params)


def register(name: str, factory: Callable[..., BaseForecaster]) -> None:
    _REGISTRY[name] = factory
