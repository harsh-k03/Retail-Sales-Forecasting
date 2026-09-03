"""Optuna hyperparameter search over the walk-forward folds."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.config import Config
from src.logger import get_logger
from src.models.train import cross_validate
from src.validation.splitters import Fold, make_splits

logger = get_logger(__name__)


def _suggest(trial, name: str, spec: Dict[str, Any]):
    kind = spec.get("type", "float")
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    return trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))


def build_objective(
    frame: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    config: Config,
    folds: List[Fold],
    metric: str = "rmse",
):
    """Reusable objective: same CV protocol as the final evaluation."""
    space = config.search_space(model_name)
    if not space:
        raise ValueError(f"No search space configured for '{model_name}'")

    def objective(trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        result = cross_validate(frame, features, model_name, config, folds, **params)
        if not result.ok:
            raise RuntimeError(result.error)
        return float(result.cv_metrics.get(metric, float("inf")))

    return objective


def tune_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    config: Optional[Config] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = config or Config.load()
    folds = make_splits(frame, config)
    metric = config.get("training.primary_metric", "rmse")

    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_name}-{metric}",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    study.optimize(
        build_objective(frame, features, model_name, config, folds, metric),
        n_trials=n_trials or int(config.get("tuning.n_trials", 25)),
        timeout=timeout or int(config.get("tuning.timeout", 600)),
        show_progress_bar=False,
    )
    logger.info("Best %s %s=%.4f params=%s", model_name, metric, study.best_value, study.best_params)
    return {
        "model": model_name,
        "metric": metric,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }


def tune_all(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: Optional[Config] = None,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    config = config or Config.load()
    names = list(models or config.get("tuning.models", []))
    return {name: tune_model(frame, features, name, config) for name in names}
