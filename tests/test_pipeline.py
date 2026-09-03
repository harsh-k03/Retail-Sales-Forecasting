"""End-to-end pipeline integration."""
from __future__ import annotations

import json

import pytest

from src.pipeline import Pipeline

FAST_MODELS = ["naive", "lightgbm"]


@pytest.fixture(scope="module")
def artifacts(raw_frame, tmp_path_factory):
    from src.config import Config

    config = Config.load()
    config.root = tmp_path_factory.mktemp("run")
    config.data["split"]["n_splits"] = 2
    config.data["split"]["test_size_days"] = 14
    config.data["explain"]["max_samples"] = 400
    pipeline = Pipeline(config)
    return pipeline.run(raw_frame, models=FAST_MODELS, horizon=5, preset="favorita", make_figures=True), config


@pytest.mark.slow
def test_pipeline_produces_every_artifact(artifacts):
    result, _ = artifacts
    assert result.validation.passed
    assert result.clean is not None and len(result.clean) > 0
    assert result.features is not None
    assert result.comparison is not None and len(result.comparison) == len(FAST_MODELS)
    assert result.best_model in FAST_MODELS
    assert result.bundle is not None
    assert result.forecast is not None and len(result.forecast) > 0
    assert not result.insights.empty


@pytest.mark.slow
def test_pipeline_writes_reports_to_disk(artifacts):
    _, config = artifacts
    metrics = config.path("metrics")
    for name in ("model_comparison.csv", "run_summary.json", "insights.csv"):
        assert (metrics / name).exists()
    summary = json.loads((metrics / "run_summary.json").read_text())
    assert summary["best_model"] in FAST_MODELS
    assert summary["n_features"] > 20
    assert (config.path("validation") / "validation_report.json").exists()
    assert (config.path("figures") / "trend.png").exists()


@pytest.mark.slow
def test_saved_bundle_can_forecast_again(artifacts):
    from src.forecast.recursive import forecast
    from src.models.persistence import latest_bundle, load_bundle

    result, config = artifacts
    bundle = load_bundle(latest_bundle(config.path("models")))
    out = forecast(bundle.model, result.clean, bundle.builder, horizon=3, config=config)
    assert len(out) > 0 and (out["forecast"] >= 0).all()


def test_validation_failure_stops_the_run(raw_frame, tmp_path):
    from src.config import Config

    config = Config.load()
    config.root = tmp_path
    broken = raw_frame.drop(columns=["family"])
    with pytest.raises(ValueError, match="Validation failed"):
        Pipeline(config).run(broken, models=["naive"], preset="favorita", make_figures=False)
