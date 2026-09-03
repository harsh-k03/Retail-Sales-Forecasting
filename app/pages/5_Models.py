"""Train and compare forecasting models with walk-forward validation."""
from __future__ import annotations

import streamlit as st

from app.components import charts, state
from src.models.persistence import ModelBundle, bundle_path, save_bundle
from src.models.registry import available_models, is_available
from src.models.train import best_model_name, comparison_table, fit_final_model, train_models

state.set_page("Models", "🤖")
if not state.require_data():
    st.stop()
if not state.ensure_features():
    st.stop()

config = state.get_config()
features, builder = state.get("features"), state.get("builder")
installed = [m for m in available_models() if is_available(m)]

with st.sidebar:
    st.header("Training")
    picked = st.multiselect("Models", installed, default=[m for m in config.get("training.models", []) if m in installed])
    n_splits = st.slider("Walk-forward folds", 2, 8, int(config.get("split.n_splits", 4)))
    test_days = st.slider("Test window (days)", 7, 90, int(config.get("split.test_size_days", 30)))
    run = st.button("Train and compare", type="primary", use_container_width=True)

config.data.setdefault("split", {})
config.data["split"]["n_splits"] = n_splits
config.data["split"]["test_size_days"] = test_days

if run and picked:
    with st.spinner("Walk-forward training..."):
        results = train_models(features, builder.spec.feature_names, config, models=picked)
        table = comparison_table(results, config.get("training.primary_metric", "rmse"))
    state.put("results", results)
    state.put("comparison", table)

table = state.get("comparison")
if table is None:
    st.info("Pick the models to compare in the sidebar, then train.")
    st.stop()

st.subheader("Comparison (mean across folds)")
st.dataframe(table, use_container_width=True, hide_index=True)

ok = table[table["status"] == "ok"]
if not ok.empty:
    st.plotly_chart(charts.bar(ok, "model", "rmse", "RMSE by model (lower is better)", horizontal=True),
                    use_container_width=True)

    best = best_model_name(table, config.get("training.primary_metric", "rmse"))
    st.success(f"Best model: **{best}**")

    results = state.get("results")
    chosen = st.selectbox("Inspect model", list(results), index=list(results).index(best))
    result = results[chosen]
    if result.ok and result.predictions is not None:
        left, right = st.columns(2)
        left.plotly_chart(charts.actual_vs_predicted(result.predictions), use_container_width=True)
        right.plotly_chart(charts.residual_plot(result.predictions), use_container_width=True)
        st.dataframe(result.fold_metrics, use_container_width=True, hide_index=True)

    if st.button("Refit best model on all data and save", type="primary"):
        with st.spinner("Refitting..."):
            model = fit_final_model(features, builder.spec.feature_names, best, config)
            bundle = ModelBundle(
                model=model, builder=builder, feature_names=builder.spec.feature_names,
                model_name=best, metrics={k: v for k, v in results[best].cv_metrics.items() if isinstance(v, float)},
                metadata={"rows_trained": len(features)},
            )
            save_bundle(bundle, bundle_path(config.path("models"), best))
        state.put("bundle", bundle)
        state.put("forecast", None)
        st.success(f"Saved model bundle for {best}.")
