#!/usr/bin/env python3
"""CLI entry point for the end-to-end pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, available_schemas  # noqa: E402
from src.logger import configure_logging, get_logger  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retail sales forecasting pipeline")
    parser.add_argument("--data", default=None, help="CSV/parquet path (default: config data.sample_file)")
    parser.add_argument("--preset", default=None, choices=available_schemas(), help="Dataset schema preset")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to compare")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in days")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    parser.add_argument("--tune", action="store_true", help="Run Optuna before final training")
    parser.add_argument("--config", default=None, help="Alternative config.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config.load(config_path=args.config)
    configure_logging(
        level=config.get("logging.level", "INFO"),
        log_file=config.get("logging.file", "logs/app.log"),
    )
    logger = get_logger("run_pipeline")

    source = args.data or config.get("data.sample_file")
    pipeline = Pipeline(config)

    if args.tune:
        from src.models.tuning import tune_all

        pipeline.ingest(source, preset=args.preset)
        pipeline.validate()
        pipeline.clean()
        pipeline.engineer()
        best = tune_all(pipeline.artifacts.features, pipeline.artifacts.builder.spec.feature_names, config)
        logger.info("Tuning finished: %s", {k: v["best_params"] for k, v in best.items()})
        params = {k: v["best_params"] for k, v in best.items()}
        pipeline.explore(make_figures=not args.no_figures)
        pipeline.train(models=args.models, params=params)
        pipeline.finalize()
        pipeline.explain()
        pipeline.forecast(args.horizon)
        pipeline.insights()
        artifacts = pipeline.artifacts
    else:
        artifacts = pipeline.run(
            source,
            models=args.models,
            horizon=args.horizon,
            preset=args.preset,
            make_figures=not args.no_figures,
        )

    print("\n=== Model comparison ===")
    print(artifacts.comparison.to_string(index=False))
    print(f"\nBest model: {artifacts.best_model}")
    print(f"Reports: {config.path('metrics')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
