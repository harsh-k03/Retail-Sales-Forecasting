"""Walk-forward training and model comparison."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import Config
from src.data import schema as S
from src.logger import get_logger
from src.models import metrics as M
from src.models.base import BaseForecaster
from src.models.registry import create_model, is_available
from src.validation.splitters import Fold, make_splits

logger = get_logger(__name__)

CONTEXT_COLUMNS = [S.DATE, S.STORE, S.PRODUCT, "promotion", "holiday"]


@dataclass
class TrainingResult:
    model_name: str
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    holdout_metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[pd.DataFrame] = None
    importance: Optional[pd.DataFrame] = None
    model: Optional[BaseForecaster] = None
    fit_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.model is not None


def _context(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[c for c in CONTEXT_COLUMNS if c in frame.columns]]


def _fit_predict(model: BaseForecaster, frame: pd.DataFrame, features: Sequence[str],
                 train_idx, test_idx, target: str):
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    model.fit(train[list(features)], train[target], context=_context(train))
    preds = model.predict(test[list(features)], context=_context(test))
    return test, np.clip(np.asarray(preds, dtype=float), 0, None)


def cross_validate(
    frame: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    config: Optional[Config] = None,
    folds: Optional[List[Fold]] = None,
    **overrides,
) -> TrainingResult:
    """Walk-forward CV for a single model."""
    config = config or Config.load()
    target = config.get("training.target", S.TARGET)
    folds = folds if folds is not None else make_splits(frame, config)
    result = TrainingResult(model_name=model_name)
    started = time.perf_counter()
    predictions: List[pd.DataFrame] = []

    try:
        for fold in folds:
            model = create_model(model_name, config, **overrides)
            test, preds = _fit_predict(model, frame, features, fold.train_idx, fold.test_idx, target)
            scores = M.evaluate(test[target], preds)
            scores["fold"] = fold.index
            result.fold_metrics.append(scores)
            block = test[[c for c in (S.DATE, S.STORE, S.PRODUCT) if c in test.columns]].copy()
            block["actual"] = test[target].to_numpy()
            block["prediction"] = preds
            block["fold"] = fold.index
            predictions.append(block)
            logger.info("%s fold %d: RMSE=%.3f MAPE=%.2f%%", model_name, fold.index, scores["rmse"], scores["mape"])

        result.cv_metrics = M.summarize_folds(result.fold_metrics)
        result.holdout_metrics = result.fold_metrics[-1] if result.fold_metrics else {}
        result.predictions = pd.concat(predictions, ignore_index=True) if predictions else None
        result.model = model
        result.importance = model.feature_importance()
    except Exception as exc:  # keep the comparison running if one model fails
        logger.exception("Model %s failed: %s", model_name, exc)
        result.error = f"{type(exc).__name__}: {exc}"
    result.fit_seconds = time.perf_counter() - started
    return result


def fit_final_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    config: Optional[Config] = None,
    **overrides,
) -> BaseForecaster:
    """Refit on the full history before forecasting the future."""
    config = config or Config.load()
    target = config.get("training.target", S.TARGET)
    model = create_model(model_name, config, **overrides)
    model.fit(frame[list(features)], frame[target], context=_context(frame))
    return model


def train_models(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: Optional[Config] = None,
    models: Optional[Sequence[str]] = None,
    params: Optional[Dict[str, Dict]] = None,
) -> Dict[str, TrainingResult]:
    config = config or Config.load()
    names = list(models or config.get("training.models", ["naive", "lightgbm"]))
    params = params or {}
    folds = make_splits(frame, config)
    logger.info("Walk-forward folds: %s", [f.describe() for f in folds])

    results: Dict[str, TrainingResult] = {}
    for name in names:
        if not is_available(name):
            logger.warning("Skipping %s: dependency not installed", name)
            continue
        results[name] = cross_validate(frame, features, name, config, folds, **params.get(name, {}))
    return results


def comparison_table(results: Dict[str, TrainingResult], primary: str = "rmse") -> pd.DataFrame:
    """One row per model, ranked by the primary metric, with skill vs naive."""
    rows = []
    for name, res in results.items():
        if not res.ok:
            rows.append({"model": name, "status": "failed", "error": res.error})
            continue
        row = {"model": name, "status": "ok", "fit_seconds": round(res.fit_seconds, 2)}
        for key in ("rmse", "mae", "mape", "smape", "r2", "bias"):
            row[key] = res.cv_metrics.get(key, float("nan"))
        row["rmse_std"] = res.cv_metrics.get("rmse_std", float("nan"))
        rows.append(row)

    table = pd.DataFrame(rows)
    if "rmse" in table.columns:
        baseline = table.loc[table["model"] == "naive", primary]
        base_value = float(baseline.iloc[0]) if len(baseline) else None
        table["skill_vs_naive_pct"] = table[primary].apply(lambda v: M.skill_score(v, base_value))
        table = table.sort_values(primary, na_position="last").reset_index(drop=True)
    return table


def best_model_name(table: pd.DataFrame, primary: str = "rmse") -> str:
    ok = table[table.get("status", "ok") == "ok"]
    if ok.empty:
        raise RuntimeError("No model trained successfully")
    return str(ok.sort_values(primary).iloc[0]["model"])
