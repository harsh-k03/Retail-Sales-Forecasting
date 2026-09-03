"""Recursive forecasting and scenario simulation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import schema as S
from src.forecast.recursive import aggregate_forecast, forecast, forecast_summary, make_future_frame
from src.insights.scenario import Scenario, ScenarioSimulator
from src.models.train import fit_final_model


@pytest.fixture(scope="module")
def fitted(feature_frame, builder, config):
    return fit_final_model(feature_frame, builder.spec.feature_names, "lightgbm", config)


def test_forecast_covers_every_series_and_step(fitted, clean_frame, builder, config):
    horizon = 5
    out = forecast(fitted, clean_frame, builder, horizon=horizon, config=config)
    n_series = clean_frame.groupby([S.STORE, S.PRODUCT], observed=True).ngroups
    assert len(out) == n_series * horizon
    assert out["step"].max() == horizon


def test_forecast_dates_start_after_history(fitted, clean_frame, builder, config):
    out = forecast(fitted, clean_frame, builder, horizon=3, config=config)
    assert out[S.DATE].min() == clean_frame[S.DATE].max() + pd.Timedelta(days=1)


def test_forecast_is_non_negative(fitted, clean_frame, builder, config):
    out = forecast(fitted, clean_frame, builder, horizon=5, config=config)
    assert (out["forecast"] >= 0).all()
    assert out["forecast"].notna().all()


def test_forecast_magnitude_is_plausible(fitted, clean_frame, builder, config):
    out = forecast(fitted, clean_frame, builder, horizon=7, config=config)
    recent_mean = clean_frame[S.TARGET].tail(500).mean()
    assert 0.4 * recent_mean < out["forecast"].mean() < 2.0 * recent_mean


def test_overrides_are_applied_to_the_future_frame(clean_frame):
    future = make_future_frame(clean_frame, 4, overrides={"promotion": 1})
    assert (future["promotion"] == 1).all()


def test_summary_and_aggregation(fitted, clean_frame, builder, config):
    out = forecast(fitted, clean_frame, builder, horizon=5, config=config)
    summary = forecast_summary(out, clean_frame)
    assert summary["horizon_days"] == 5
    assert summary["total_forecast"] > 0
    assert len(aggregate_forecast(out)) == 5


def test_scenario_changes_the_total(fitted, clean_frame, builder, config):
    simulator = ScenarioSimulator(fitted, clean_frame, builder, config)
    result = simulator.compare(Scenario("promo", {"promotion": 1}, horizon=5))
    assert result["baseline_total"] > 0
    assert result["scenario_total"] != result["baseline_total"]


def test_unsupported_lever_is_ignored(fitted, clean_frame, builder, config):
    simulator = ScenarioSimulator(fitted, clean_frame, builder, config)
    out = simulator.run(Scenario("bogus", {"not_a_lever": 5}, horizon=3))
    assert len(out) > 0
