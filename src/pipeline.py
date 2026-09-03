"""End-to-end orchestration: raw file in, trained model + reports out."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

from src.config import Config
from src.data import schema as S
from src.data.cleaning import clean
from src.data.loader import load_dataset
from src.data.validation import ValidationReport, validate
from src.eda import analysis as eda_analysis
from src.eda import plots as eda_plots
from src.explain import shap_explainer
from src.features.builder import FeatureBuilder
from src.forecast.recursive import forecast as run_forecast
from src.forecast.recursive import forecast_summary
from src.insights.engine import InsightEngine
from src.logger import get_logger
from src.models.persistence import ModelBundle, bundle_path, save_bundle, write_metadata
from src.models.train import TrainingResult, best_model_name, comparison_table, fit_final_model, train_models
from src.utils import save_json, set_seed, timed

logger = get_logger(__name__)


@dataclass
class PipelineArtifacts:
    """Everything a pipeline run produced, for the report and the dashboard."""

    raw: Optional[pd.DataFrame] = None
    clean: Optional[pd.DataFrame] = None
    features: Optional[pd.DataFrame] = None
    builder: Optional[FeatureBuilder] = None
    validation: Optional[ValidationReport] = None
    cleaning_stats: Dict[str, int] = field(default_factory=dict)
    eda: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, TrainingResult] = field(default_factory=dict)
    comparison: Optional[pd.DataFrame] = None
    best_model: Optional[str] = None
    bundle: Optional[ModelBundle] = None
    explanation: Optional[Any] = None
    forecast: Optional[pd.DataFrame] = None
    insights: Optional[pd.DataFrame] = None
    figures: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Each stage is callable on its own; `run` wires them together."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        set_seed(self.config.seed)
        self.artifacts = PipelineArtifacts()

    # ---------------------------------------------------------------- stages
    def ingest(self, source: Union[str, Path, pd.DataFrame], preset: Optional[str] = None) -> pd.DataFrame:
        frame = load_dataset(source, self.config, preset=preset)
        self.artifacts.raw = frame
        return frame

    def validate(self, frame: Optional[pd.DataFrame] = None) -> ValidationReport:
        frame = frame if frame is not None else self.artifacts.raw
        report = validate(frame, self.config)
        report.save(self.config.path("validation") / "validation_report.json")
        self.artifacts.validation = report
        return report

    def clean(self, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        frame = frame if frame is not None else self.artifacts.raw
        cleaned, stats = clean(frame, self.config)
        self.artifacts.clean, self.artifacts.cleaning_stats = cleaned, stats
        return cleaned

    def explore(self, frame: Optional[pd.DataFrame] = None, make_figures: bool = True) -> Dict[str, Any]:
        frame = frame if frame is not None else self.artifacts.clean
        result = eda_analysis.run_eda(frame)
        self.artifacts.eda = result
        if make_figures:
            self.artifacts.figures.update(eda_plots.generate_all(frame, self.config.path("figures")))
        return result

    def engineer(self, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        frame = frame if frame is not None else self.artifacts.clean
        builder = FeatureBuilder(self.config)
        features = builder.drop_warmup(builder.fit_transform(frame))
        self.artifacts.features, self.artifacts.builder = features, builder
        return features

    def train(self, models: Optional[Sequence[str]] = None, params: Optional[Dict] = None) -> pd.DataFrame:
        features = self.artifacts.features
        names = self.artifacts.builder.spec.feature_names
        results = train_models(features, names, self.config, models=models, params=params)
        table = comparison_table(results, self.config.get("training.primary_metric", "rmse"))
        self.artifacts.results, self.artifacts.comparison = results, table
        self.artifacts.best_model = best_model_name(table, self.config.get("training.primary_metric", "rmse"))
        logger.info("Best model: %s", self.artifacts.best_model)
        return table

    def finalize(self, model_name: Optional[str] = None, params: Optional[Dict] = None) -> ModelBundle:
        """Refit the winner on all history and persist the bundle."""
        model_name = model_name or self.artifacts.best_model
        builder = self.artifacts.builder
        model = fit_final_model(
            self.artifacts.features, builder.spec.feature_names, model_name, self.config, **(params or {})
        )
        result = self.artifacts.results.get(model_name)
        bundle = ModelBundle(
            model=model,
            builder=builder,
            feature_names=builder.spec.feature_names,
            target=self.config.get("training.target", S.TARGET),
            model_name=model_name,
            metrics={k: v for k, v in (result.cv_metrics if result else {}).items() if isinstance(v, float)},
            metadata={
                "schema_preset": self.config.get("data.schema_preset"),
                "rows_trained": int(len(self.artifacts.features)),
                "date_max": str(self.artifacts.clean[S.DATE].max().date()),
                "n_stores": int(self.artifacts.clean[S.STORE].nunique()),
                "n_products": int(self.artifacts.clean[S.PRODUCT].nunique()),
            },
        )
        save_bundle(bundle, bundle_path(self.config.path("models"), model_name))
        write_metadata(bundle, self.config.path("metrics") / "model_metadata.json")
        self.artifacts.bundle = bundle
        return bundle

    def _shap_model(self):
        """The winner if it supports SHAP, else the best tree model that does."""
        bundle = self.artifacts.bundle
        if bundle is not None and getattr(bundle.model, "supports_shap", False):
            return bundle.model, bundle.model_name
        table = self.artifacts.comparison
        if table is None:
            return None, None
        for name in table.loc[table["status"] == "ok", "model"]:
            result = self.artifacts.results.get(name)
            if result and result.model is not None and getattr(result.model, "supports_shap", False):
                logger.info("%s has no SHAP support; explaining %s instead", bundle.model_name if bundle else "?", name)
                return result.model, name
        return None, None

    def explain(self) -> Optional[Any]:
        if not self.config.get("explain.enabled", True) or self.artifacts.bundle is None:
            return None
        model, explained_name = self._shap_model()
        if model is None:
            logger.info("No SHAP-capable model available")
            return None
        self.artifacts.summary["explained_model"] = explained_name
        exp = shap_explainer.explain(
            model,
            self.artifacts.features[self.artifacts.builder.spec.feature_names],
            max_samples=int(self.config.get("explain.max_samples", 2000)),
            seed=self.config.seed,
        )
        if exp is not None:
            self.artifacts.figures.update(
                shap_explainer.generate_all(exp, self.config.path("figures"), int(self.config.get("explain.top_features", 20)))
            )
            exp.global_importance().to_csv(self.config.path("metrics") / "shap_importance.csv", index=False)
        self.artifacts.explanation = exp
        return exp

    def forecast(self, horizon: Optional[int] = None) -> pd.DataFrame:
        horizon = horizon or int(self.config.get("forecast.default_horizon", 30))
        frame = run_forecast(
            self.artifacts.bundle.model, self.artifacts.clean, self.artifacts.builder,
            horizon=horizon, config=self.config,
        )
        frame.to_csv(self.config.path("metrics") / f"forecast_{horizon}d.csv", index=False)
        self.artifacts.forecast = frame
        return frame

    def insights(self) -> pd.DataFrame:
        importance = None
        if self.artifacts.explanation is not None:
            importance = self.artifacts.explanation.global_importance()
        elif self.artifacts.bundle is not None:
            importance = self.artifacts.bundle.model.feature_importance()
        engine = InsightEngine(self.artifacts.clean, self.artifacts.forecast, importance)
        table = engine.to_frame()
        table.to_csv(self.config.path("metrics") / "insights.csv", index=False)
        self.artifacts.insights = table
        return table

    # ------------------------------------------------------------------- run
    def run(
        self,
        source: Union[str, Path, pd.DataFrame],
        models: Optional[Sequence[str]] = None,
        horizon: Optional[int] = None,
        preset: Optional[str] = None,
        make_figures: bool = True,
    ) -> PipelineArtifacts:
        with timed("ingest"):
            self.ingest(source, preset=preset)
        report = self.validate()
        if not report.passed:
            raise ValueError(f"Validation failed: {[i.message for i in report.errors]}")
        with timed("clean"):
            self.clean()
        with timed("eda"):
            self.explore(make_figures=make_figures)
        with timed("features"):
            self.engineer()
        with timed("train"):
            table = self.train(models=models)
        with timed("finalize"):
            self.finalize()
        with timed("explain"):
            self.explain()
        with timed("forecast"):
            forecast_frame = self.forecast(horizon)
        with timed("insights"):
            self.insights()

        if make_figures:
            figures_dir = self.config.path("figures")
            self.artifacts.figures["forecast"] = str(
                eda_plots.plot_forecast(self.artifacts.clean, forecast_frame, figures_dir)
            )
            ok = table[table["status"] == "ok"]
            if not ok.empty:
                self.artifacts.figures["model_comparison"] = str(
                    eda_plots.plot_model_comparison(ok, figures_dir)
                )

        self._write_reports(forecast_frame)
        return self.artifacts

    def _write_reports(self, forecast_frame: pd.DataFrame) -> None:
        metrics_dir = self.config.path("metrics")
        self.artifacts.comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
        for name, result in self.artifacts.results.items():
            if result.predictions is not None:
                result.predictions.to_csv(metrics_dir / f"predictions_{name}.csv", index=False)

        summary: Dict[str, Any] = {
            "dataset": self.artifacts.validation.summary if self.artifacts.validation else {},
            "cleaning": self.artifacts.cleaning_stats,
            "kpis": self.artifacts.eda.get("kpis", {}),
            "best_model": self.artifacts.best_model,
            "metrics": self.artifacts.bundle.metrics if self.artifacts.bundle else {},
            "forecast": forecast_summary(forecast_frame, self.artifacts.clean),
            "explained_model": self.artifacts.summary.get("explained_model"),
            "n_features": len(self.artifacts.builder.spec.feature_names),
            "figures": self.artifacts.figures,
        }
        save_json(summary, metrics_dir / "run_summary.json")
        self.artifacts.summary = summary
        logger.info("Reports written to %s", metrics_dir)


def run_pipeline(source: Union[str, Path, pd.DataFrame], **kwargs) -> PipelineArtifacts:
    return Pipeline(kwargs.pop("config", None)).run(source, **kwargs)
