"""Session-state helpers shared by every dashboard page."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import Config  # noqa: E402
from src.data.cleaning import clean  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.validation import validate  # noqa: E402
from src.features.builder import FeatureBuilder  # noqa: E402
from src.models.persistence import latest_bundle, load_bundle  # noqa: E402

DEFAULTS = {
    "raw": None,
    "clean": None,
    "features": None,
    "builder": None,
    "validation": None,
    "cleaning_stats": {},
    "results": {},
    "comparison": None,
    "bundle": None,
    "forecast": None,
    "explanation": None,
    "source_name": None,
}


def get_config() -> Config:
    if "config" not in st.session_state:
        st.session_state["config"] = Config.load()
    return st.session_state["config"]


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def set_page(title: str, icon: str = "📈") -> None:
    st.set_page_config(page_title=f"{title} | Retail Forecasting", page_icon=icon, layout="wide")
    init_state()
    st.title(f"{icon} {title}")


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def put(key: str, value: Any) -> None:
    st.session_state[key] = value


def load_source(source, preset: Optional[str] = None, name: str = "dataset") -> pd.DataFrame:
    """Load, validate and clean a source into session state."""
    config = get_config()
    raw = load_dataset(source, config, preset=preset)
    report = validate(raw, config)
    put("raw", raw)
    put("validation", report)
    put("source_name", name)
    if report.passed:
        cleaned, stats = clean(raw, config)
        put("clean", cleaned)
        put("cleaning_stats", stats)
    else:
        put("clean", None)
    for key in ("features", "builder", "results", "comparison", "forecast", "explanation"):
        put(key, DEFAULTS[key])
    return raw


def load_sample() -> pd.DataFrame:
    config = get_config()
    return load_source(config.get("data.sample_file"), name="sample_store_sales.csv")


def ensure_features() -> bool:
    """Build features on demand; returns False when no data is loaded."""
    if get("clean") is None:
        return False
    if get("features") is None:
        builder = FeatureBuilder(get_config())
        features = builder.drop_warmup(builder.fit_transform(get("clean")))
        put("features", features)
        put("builder", builder)
    return True


def load_saved_model() -> Optional[Any]:
    if get("bundle") is not None:
        return get("bundle")
    path = latest_bundle(get_config().path("models"))
    if path is None:
        return None
    bundle = load_bundle(path)
    put("bundle", bundle)
    return bundle


def require_data() -> bool:
    if get("clean") is None:
        st.warning("No dataset loaded. Go to **Upload** and load a CSV or the bundled sample.")
        return False
    return True


def require_model() -> bool:
    if load_saved_model() is None:
        st.warning("No trained model found. Train one on the **Models** page or run `python scripts/run_pipeline.py`.")
        return False
    return True
