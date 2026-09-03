"""SHAP explanations for the trained model."""
from __future__ import annotations

import streamlit as st

from app.components import charts, state
from src.explain import shap_explainer

state.set_page("Explainability", "🧠")
if not (state.require_data() and state.require_model()):
    st.stop()
if not state.ensure_features():
    st.stop()

config = state.get_config()
bundle = state.get("bundle")
features = state.get("features")

if not getattr(bundle.model, "supports_shap", False):
    st.info(f"`{bundle.model_name}` does not expose SHAP values. Train a tree model to use this page.")
    st.stop()

max_samples = st.slider("Rows sampled for SHAP", 200, 5000, int(config.get("explain.max_samples", 2000)), step=200)
if st.button("Compute SHAP values", type="primary") or state.get("explanation") is not None:
    if state.get("explanation") is None:
        with st.spinner("Computing SHAP values..."):
            state.put("explanation", shap_explainer.explain(bundle.model, features[bundle.feature_names], max_samples))

explanation = state.get("explanation")
if explanation is None:
    st.info("Compute SHAP values to see global and local explanations.")
    st.stop()

st.subheader("Global importance")
importance = explanation.global_importance(int(config.get("explain.top_features", 20)))
st.plotly_chart(charts.bar(importance, "feature", "mean_abs_shap", "Mean |SHAP| by feature", horizontal=True),
                use_container_width=True)

st.subheader("Direction of effect")
st.dataframe(explanation.direction_table(15), use_container_width=True, hide_index=True)

st.subheader("Local explanation")
row = st.number_input("Sample row", 0, len(explanation.sample) - 1, 0)
st.dataframe(explanation.local(int(row)), use_container_width=True, hide_index=True)
st.caption(f"Base value (expected prediction): {explanation.base_value:,.2f}")
