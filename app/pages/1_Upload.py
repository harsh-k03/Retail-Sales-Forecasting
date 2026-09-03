"""Upload a dataset or load the bundled sample."""
from __future__ import annotations

import streamlit as st

from app.components import state
from src.config import available_schemas
from src.data.loader import infer_preset, read_table

state.set_page("Upload", "📤")
config = state.get_config()

st.write("Upload a retail sales file, or load the bundled Kaggle-schema sample.")

uploaded = st.file_uploader("CSV or Parquet", type=["csv", "parquet"])
preset_choice = st.selectbox("Schema preset", ["auto"] + available_schemas(), index=0)

col1, col2 = st.columns(2)
if col1.button("Load file", disabled=uploaded is None, use_container_width=True):
    preset = preset_choice
    if preset == "auto":
        uploaded.seek(0)
        preset = infer_preset(read_table(uploaded, nrows=5))
        st.info(f"Inferred preset: `{preset}`")
    uploaded.seek(0)
    with st.spinner("Loading and validating..."):
        state.load_source(uploaded, preset=preset, name=uploaded.name)
    st.success(f"Loaded {uploaded.name}")

if col2.button("Load bundled sample", use_container_width=True):
    with st.spinner("Loading sample..."):
        state.load_sample()
    st.success("Sample loaded")

raw = state.get("raw")
if raw is not None:
    st.divider()
    st.subheader("Preview (canonical schema)")
    st.dataframe(raw.head(50), use_container_width=True)
    st.caption(f"{len(raw):,} rows × {raw.shape[1]} columns")
    report = state.get("validation")
    if report is not None:
        if report.passed:
            st.success("Schema validation passed. See the Validation page for details.")
        else:
            st.error("Validation failed. Open the Validation page to review the issues.")
