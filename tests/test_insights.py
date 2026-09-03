"""Insight engine and EDA summaries."""
from __future__ import annotations

import pandas as pd

from src.data import schema as S
from src.eda import analysis
from src.insights.engine import InsightEngine


def test_kpi_summary_fields(clean_frame):
    kpis = analysis.kpi_summary(clean_frame)
    assert kpis["total_sales"] > 0
    assert kpis["n_stores"] == clean_frame[S.STORE].nunique()


def test_store_shares_sum_to_100(clean_frame):
    assert abs(analysis.store_performance(clean_frame)["share_pct"].sum() - 100) < 1e-6


def test_promotion_effect_reports_uplift(clean_frame):
    table = analysis.promotion_effect(clean_frame)
    assert "uplift_pct" in table.columns


def test_anomaly_detector_finds_an_injected_spike(clean_frame):
    frame = clean_frame.copy()
    spike_date = frame[S.DATE].max() - pd.Timedelta(days=10)
    frame.loc[frame[S.DATE] == spike_date, S.TARGET] *= 12
    anomalies = analysis.detect_anomalies(frame)
    assert spike_date in set(anomalies[S.DATE])


def test_insights_are_generated_and_ranked(clean_frame):
    table = InsightEngine(clean_frame).to_frame()
    assert not table.empty
    assert set(table.columns) == {"priority", "category", "title", "detail", "action"}
    order = {"high": 0, "medium": 1, "low": 2}
    ranks = [order[p] for p in table["priority"]]
    assert ranks == sorted(ranks)


def test_insights_include_a_forecast_rule(clean_frame):
    fake_forecast = pd.DataFrame(
        {
            S.DATE: pd.date_range(clean_frame[S.DATE].max() + pd.Timedelta(days=1), periods=5),
            S.STORE: clean_frame[S.STORE].iloc[0],
            S.PRODUCT: clean_frame[S.PRODUCT].iloc[0],
            "forecast": 1.0,
            "step": range(1, 6),
        }
    )
    table = InsightEngine(clean_frame, fake_forecast).to_frame()
    assert "demand" in set(table["category"])


def test_engine_survives_a_broken_rule(clean_frame, monkeypatch):
    monkeypatch.setattr(analysis, "promotion_effect", lambda *_: (_ for _ in ()).throw(ValueError("boom")))
    assert not InsightEngine(clean_frame).to_frame().empty
