# Milestone Plan

| # | Phase | Deliverable | Status |
|---|---|---|---|
| 1 | Project setup | Repo structure, config layer, logging, schema presets | ✅ |
| 2 | Data engineering | Loader with Kaggle adapters, validation rules + report, cleaning | ✅ |
| 3 | EDA | Trend, seasonality, distributions, heatmaps, anomalies, stability, figures | ✅ |
| 4 | Feature engineering | Calendar, Fourier, event, lag, rolling, interaction features with leakage guards | ✅ |
| 5 | Modelling | Naive, RandomForest, XGBoost, LightGBM, CatBoost, Prophet on shared walk-forward folds | ✅ |
| 6 | Optimisation | Optuna objectives and persisted best configurations | ✅ |
| 7 | Explainability & insights | SHAP global/local, business insight engine, scenario simulator | ✅ |
| 8 | Application | Streamlit dashboard (11 pages) + FastAPI service | ✅ |
| 9 | Deployment & docs | Docker, compose, CI, tests, README and design docs | ✅ |

## Stretch goals

- [ ] MLflow experiment tracking
- [ ] Prediction intervals (quantile regression or conformal)
- [ ] Database-backed prediction history
- [ ] Authentication on the API
- [ ] LSTM / Temporal Fusion Transformer comparison
- [ ] Scheduled retraining with drift monitoring
