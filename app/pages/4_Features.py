"""Feature engineering inspection."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import state

state.set_page("Feature Engineering", "🧬")
if not state.require_data():
    st.stop()

config = state.get_config()
st.write("Features are rebuilt from config. Every lag and rolling window is shifted, so no row sees its own target.")

with st.expander("Active feature settings", expanded=False):
    st.json(config.get("features", {}))

if st.button("Build features", type="primary"):
    with st.spinner("Engineering features..."):
        state.put("features", None)
        state.ensure_features()

if not state.ensure_features():
    st.stop()

features = state.get("features")
builder = state.get("builder")

col1, col2, col3 = st.columns(3)
col1.metric("Feature count", len(builder.spec.feature_names))
col2.metric("Rows after warm-up", f"{len(features):,}")
col3.metric("Warm-up rows dropped", builder.spec.warmup_rows)

groups = {
    "calendar": [c for c in builder.spec.feature_names if c in
                 {"year", "month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter",
                  "is_weekend", "is_month_start", "is_month_end", "is_quarter_end", "days_in_month", "is_payday_window"}],
    "fourier": [c for c in builder.spec.feature_names if c.startswith(("sin_", "cos_"))],
    "lag / rolling": [c for c in builder.spec.feature_names if c.startswith("sales_")],
    "events": [c for c in builder.spec.feature_names if c.startswith(("promo", "holiday"))],
    "encoded ids": builder.spec.categorical_codes,
}
st.subheader("Feature groups")
st.dataframe(
    pd.DataFrame([{"group": k, "count": len(v), "examples": ", ".join(v[:5])} for k, v in groups.items()]),
    use_container_width=True, hide_index=True,
)

st.subheader("Sample of the modelling matrix")
st.dataframe(features[["date", "store", "product", "sales"] + builder.spec.feature_names].head(30),
             use_container_width=True)

st.subheader("Correlation with the target")
numeric = features[builder.spec.feature_names + ["sales"]].select_dtypes("number")
corr = numeric.corr()["sales"].drop("sales").sort_values(key=abs, ascending=False).head(20)
st.dataframe(corr.reset_index().rename(columns={"index": "feature", "sales": "correlation"}),
             use_container_width=True, hide_index=True)
