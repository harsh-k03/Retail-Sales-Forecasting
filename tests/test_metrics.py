"""Metric correctness and edge cases."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import metrics as M


def test_perfect_prediction_scores():
    y = [1.0, 2.0, 3.0]
    assert M.rmse(y, y) == 0
    assert M.mae(y, y) == 0
    assert M.mape(y, y) == 0
    assert M.r2(y, y) == 1


def test_rmse_penalises_large_errors_more_than_mae():
    y_true, y_pred = [0, 0, 0, 0], [0, 0, 0, 8]
    assert M.rmse(y_true, y_pred) > M.mae(y_true, y_pred)


def test_mape_ignores_zero_actuals():
    assert np.isfinite(M.mape([0, 100], [50, 110]))
    assert np.isnan(M.mape([0, 0], [1, 2]))


def test_bias_sign():
    assert M.bias([10, 10], [12, 12]) > 0
    assert M.bias([10, 10], [8, 8]) < 0


def test_nan_values_are_ignored():
    assert M.rmse([1, np.nan, 3], [1, 5, 3]) == 0


def test_evaluate_returns_all_metrics():
    assert set(M.evaluate([1, 2], [1, 2])) == {"rmse", "mae", "mape", "smape", "r2", "bias"}


def test_summarize_folds_reports_mean_and_std():
    summary = M.summarize_folds([{"rmse": 2.0}, {"rmse": 4.0}])
    assert summary["rmse"] == 3.0
    assert summary["rmse_std"] == 1.0


def test_skill_score_against_baseline():
    assert M.skill_score(5.0, 10.0) == 50.0
    assert M.skill_score(5.0, None) is None


def test_evaluate_by_group():
    frame = pd.DataFrame({"store": ["a", "a", "b", "b"], "actual": [1, 2, 3, 4], "prediction": [1, 2, 4, 5]})
    out = M.evaluate_by_group(frame, "store")
    assert len(out) == 2
    assert out.iloc[0]["store"] == "a"
