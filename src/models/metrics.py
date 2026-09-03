"""Forecast accuracy metrics."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

EPS = 1e-9


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def mape(y_true, y_pred) -> float:
    """Percentage error over non-zero actuals only (zero-sales days are excluded)."""
    y_true, y_pred = _clean(y_true, y_pred)
    mask = np.abs(y_true) > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + EPS
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100) if len(y_true) else float("nan")


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + EPS))


def bias(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(y_pred - y_true)) if len(y_true) else float("nan")


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def evaluate_by_group(
    frame: pd.DataFrame,
    group: str,
    y_true_col: str = "actual",
    y_pred_col: str = "prediction",
) -> pd.DataFrame:
    rows = []
    for key, chunk in frame.groupby(group, observed=True):
        scores = evaluate(chunk[y_true_col], chunk[y_pred_col])
        scores[group] = key
        rows.append(scores)
    cols = [group, "rmse", "mae", "mape", "smape", "r2", "bias"]
    return pd.DataFrame(rows)[cols].sort_values("rmse")


def summarize_folds(fold_metrics: list[Dict[str, float]]) -> Dict[str, float]:
    """Mean +/- std across walk-forward folds."""
    if not fold_metrics:
        return {}
    frame = pd.DataFrame(fold_metrics)
    numeric = frame.select_dtypes("number")
    out: Dict[str, float] = {c: float(numeric[c].mean()) for c in numeric.columns}
    out.update({f"{c}_std": float(numeric[c].std(ddof=0)) for c in numeric.columns})
    return out


def skill_score(model_metric: float, baseline_metric: Optional[float]) -> Optional[float]:
    """Percent improvement over the naive baseline."""
    if baseline_metric in (None, 0) or baseline_metric != baseline_metric:
        return None
    return float((1 - model_metric / baseline_metric) * 100)
