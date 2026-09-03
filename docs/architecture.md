# Architecture

## Layers

```
                       ┌──────────────────────────────┐
                       │  config/config.yaml          │
                       │  config/schemas/*.yaml       │
                       └──────────────┬───────────────┘
                                      │ read by every layer
 raw file ──► src/data ──► src/features ──► src/models ──► src/forecast ──► src/insights
              loader          builder         registry        recursive        engine
              validation      calendar        train           horizons         scenario
              cleaning        lags            tuning
              schema          encoders        persistence
                                      │
                            src/explain (SHAP)
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                 app/streamlit_app.py         app/api/main.py
                 app/pages/*                  app/api/routers/*
```

`src/pipeline.py` wires the stages together; each stage is also callable on its own,
which is what the dashboard does when a user only wants to re-run one step.

## Data flow

1. **Ingest** — `src/data/loader.py` reads CSV/Parquet/DataFrame and maps the source columns
   onto the canonical schema using a preset from `config/schemas/`.
2. **Validate** — `src/data/validation.py` runs every rule and emits a JSON report. Errors stop
   the pipeline; warnings are recorded and the run continues.
3. **Clean** — `src/data/cleaning.py` de-duplicates on `(date, store, product)`, handles missing
   targets, clips negatives, treats outliers and densifies exogenous columns.
4. **Engineer** — `src/features/builder.py` produces calendar, Fourier, event, lag, rolling,
   expanding and interaction features, plus ordinal codes for the identifiers.
5. **Validate temporally** — `src/validation/splitters.py` builds expanding walk-forward folds.
   Nothing in this project ever shuffles rows.
6. **Train and compare** — `src/models/` fits every configured model on identical folds and
   ranks them by RMSE with a skill score against the naive baseline.
7. **Persist** — the winner is refit on all history and stored with its feature contract
   (`src/models/persistence.py`), so inference can never drift from training.
8. **Explain** — `src/explain/shap_explainer.py` computes global and local SHAP attributions.
9. **Forecast** — `src/forecast/recursive.py` predicts step by step, feeding each prediction
   back in as the next step's lag.
10. **Advise** — `src/insights/engine.py` converts history + forecast into ranked actions;
    `src/insights/scenario.py` re-runs the forecast under modified drivers.

## Key interfaces

| Contract | Where | Why it exists |
|---|---|---|
| `BaseForecaster.fit/predict` | `src/models/base.py` | Every model is swappable; Prophet and tree models share one call site |
| `FeatureSpec` | `src/features/builder.py` | The exact column list the model was trained on, saved with the model |
| `ModelBundle` | `src/models/persistence.py` | Model + builder + features + metrics in one artifact |
| `Fold` | `src/validation/splitters.py` | Explicit train/test date boundaries, assertable in tests |
| `Insight` | `src/insights/engine.py` | Each recommendation carries its own evidence |

## Extension points

- **New dataset shape** → add a YAML file to `config/schemas/`; no code change.
- **New model** → subclass `BaseForecaster`, add it to `_REGISTRY` in `src/models/registry.py`,
  add defaults and a search space to `config/model_params.yaml`.
- **New insight** → add a `_rule` method to `InsightEngine` and list it in `run()`.
- **New feature block** → add a function in `src/features/`, call it from `FeatureBuilder._engineer`,
  and register its names in `_collect_feature_names`.
