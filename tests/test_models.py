"""Model registry, training loop and persistence."""
from __future__ import annotations

import numpy as np
import pytest

from src.data import schema as S
from src.models.persistence import ModelBundle, load_bundle, save_bundle
from src.models.registry import available_models, create_model, is_available
from src.models.train import comparison_table, cross_validate, fit_final_model, train_models

FAST_MODELS = ["naive", "lightgbm"]


def test_registry_exposes_expected_models():
    assert {"naive", "random_forest", "xgboost", "lightgbm", "catboost", "prophet"} <= set(available_models())


def test_create_model_applies_config_defaults(config):
    model = create_model("lightgbm", config)
    assert model.params["n_estimators"] == config.defaults_for("lightgbm")["n_estimators"]


def test_unknown_model_raises(config):
    with pytest.raises(KeyError):
        create_model("nope", config)


def test_naive_reproduces_the_seasonal_lag(feature_frame, builder, config):
    model = create_model("naive", config)
    X = feature_frame[builder.spec.feature_names]
    model.fit(X, feature_frame[S.TARGET])
    preds = model.predict(X)
    assert np.allclose(preds, X[f"{S.TARGET}_lag_7"].to_numpy(), equal_nan=False)


@pytest.mark.parametrize("name", FAST_MODELS)
def test_cross_validate_returns_metrics(name, feature_frame, builder, config):
    result = cross_validate(feature_frame, builder.spec.feature_names, name, config)
    assert result.ok
    assert result.cv_metrics["rmse"] > 0
    assert len(result.fold_metrics) == config.get("split.n_splits")
    assert result.predictions is not None


def test_predictions_are_non_negative(feature_frame, builder, config):
    result = cross_validate(feature_frame, builder.spec.feature_names, "lightgbm", config)
    assert (result.predictions["prediction"] >= 0).all()


def test_learned_model_beats_the_naive_baseline(feature_frame, builder, config):
    results = train_models(feature_frame, builder.spec.feature_names, config, models=FAST_MODELS)
    table = comparison_table(results)
    assert (table["status"] == "ok").all()
    best = table.iloc[0]["model"]
    assert best != "naive"


def test_comparison_table_reports_skill_vs_naive(feature_frame, builder, config):
    results = train_models(feature_frame, builder.spec.feature_names, config, models=FAST_MODELS)
    table = comparison_table(results)
    assert table.loc[table["model"] == "naive", "skill_vs_naive_pct"].iloc[0] == 0


def test_failed_model_does_not_break_the_comparison(feature_frame, builder, config, monkeypatch):
    import src.models.train as train_module

    original = train_module.create_model

    def flaky(name, cfg=None, **kwargs):
        if name == "lightgbm":
            raise RuntimeError("boom")
        return original(name, cfg, **kwargs)

    monkeypatch.setattr(train_module, "create_model", flaky)
    results = train_models(feature_frame, builder.spec.feature_names, config, models=FAST_MODELS)
    table = comparison_table(results)
    assert "failed" in table["status"].tolist()


def test_feature_importance_is_ranked(feature_frame, builder, config):
    model = fit_final_model(feature_frame, builder.spec.feature_names, "lightgbm", config)
    importance = model.feature_importance()
    assert len(importance) == len(builder.spec.feature_names)
    assert importance["importance"].is_monotonic_decreasing


@pytest.mark.skipif(not is_available("lightgbm"), reason="lightgbm not installed")
def test_bundle_roundtrip(tmp_path, feature_frame, builder, config):
    model = fit_final_model(feature_frame, builder.spec.feature_names, "lightgbm", config)
    bundle = ModelBundle(model=model, builder=builder, feature_names=builder.spec.feature_names, model_name="lightgbm")
    path = save_bundle(bundle, tmp_path / "m.joblib")
    restored = load_bundle(path)
    assert restored.model_name == "lightgbm"
    assert restored.feature_names == builder.spec.feature_names
    before = model.predict(feature_frame[builder.spec.feature_names].head(20))
    after = restored.model.predict(feature_frame[builder.spec.feature_names].head(20))
    assert np.allclose(before, after)
