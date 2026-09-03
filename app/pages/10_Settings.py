"""Runtime configuration and environment overview."""
from __future__ import annotations

import json

import streamlit as st

from app.components import state
from src.config import available_schemas
from src.models.registry import available_models, is_available

state.set_page("Settings", "⚙️")
config = state.get_config()

st.subheader("Active configuration")
st.json(config.data, expanded=False)

st.subheader("Runtime overrides")
st.caption("Overrides apply to this session only. Edit `config/config.yaml` to make them permanent.")

col1, col2 = st.columns(2)
with col1:
    preset = st.selectbox("Schema preset", available_schemas(),
                          index=available_schemas().index(config.get("data.schema_preset", "canonical")))
    horizon = st.number_input("Default forecast horizon", 1, 365, int(config.get("forecast.default_horizon", 30)))
with col2:
    n_splits = st.number_input("Walk-forward folds", 2, 10, int(config.get("split.n_splits", 4)))
    test_days = st.number_input("Test window (days)", 7, 120, int(config.get("split.test_size_days", 30)))

if st.button("Apply overrides", type="primary"):
    config.data["data"]["schema_preset"] = preset
    config.data["forecast"]["default_horizon"] = int(horizon)
    config.data["split"]["n_splits"] = int(n_splits)
    config.data["split"]["test_size_days"] = int(test_days)
    st.success("Applied for this session.")

st.divider()
st.subheader("Model availability")
st.dataframe(
    [{"model": m, "installed": is_available(m)} for m in available_models()],
    use_container_width=True, hide_index=True,
)

st.subheader("Model parameters")
st.code(json.dumps(config.model_params.get("defaults", {}), indent=2), language="json")

st.divider()
if st.button("Clear session state"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session cleared. Reload the page.")
