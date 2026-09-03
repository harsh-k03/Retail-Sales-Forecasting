"""Load raw retail files and map them onto the canonical schema."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.config import Config, load_schema
from src.data import schema as S
from src.logger import get_logger

logger = get_logger(__name__)

_BINARY_TRUE = {"1", "true", "yes", "y", "t"}


def read_table(source: Union[str, Path, "pd.DataFrame"], **kwargs: Any) -> pd.DataFrame:
    """Read a CSV/parquet path, a file-like upload, or pass through a DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    name = getattr(source, "name", str(source))
    if str(name).lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(source)
    return pd.read_csv(source, **kwargs)


def apply_schema(
    frame: pd.DataFrame,
    preset: str = "canonical",
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """Rename/derive columns so the frame matches the canonical schema."""
    spec = load_schema(preset)
    mapping: Dict[str, Any] = spec.get("columns") or {}
    constants: Dict[str, Any] = spec.get("constants") or {}

    out = pd.DataFrame(index=frame.index)
    for canonical, raw in mapping.items():
        if raw and raw in frame.columns:
            out[canonical] = frame[raw]
    for canonical, value in constants.items():
        out[canonical] = value

    # keep unmapped extras that already use canonical names
    for canonical in S.REQUIRED_COLUMNS + S.OPTIONAL_COLUMNS:
        if canonical not in out.columns and canonical in frame.columns:
            out[canonical] = frame[canonical]

    return coerce_types(out, date_format=date_format)


def coerce_types(frame: pd.DataFrame, date_format: Optional[str] = None) -> pd.DataFrame:
    out = frame.copy()
    if S.DATE in out.columns:
        out[S.DATE] = pd.to_datetime(out[S.DATE], format=date_format, errors="coerce")
    for col in (S.STORE, S.PRODUCT, "region", "category"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    for col in ("sales", "temperature", "inventory", "marketing_spend"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("promotion", "holiday"):
        if col in out.columns:
            out[col] = _to_binary(out[col])
    return out


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype("int8")
    text = series.astype("string").str.lower().str.strip()
    known = text.isin(_BINARY_TRUE)
    # non-empty, non-"0"/"none" strings count as an event flag (e.g. StateHoliday='a')
    other = ~text.isin({"0", "none", "nan", "", "false", "no", "n"}) & text.notna()
    return (known | other).astype("int8")


def load_dataset(
    source: Union[str, Path, pd.DataFrame],
    config: Optional[Config] = None,
    preset: Optional[str] = None,
) -> pd.DataFrame:
    """Read a source and return it in the canonical schema, sorted by key."""
    config = config or Config.load()
    preset = preset or config.get("data.schema_preset", "canonical")
    raw = read_table(source)
    logger.info("Loaded %d rows x %d cols from %s", len(raw), raw.shape[1], source)
    mapped = apply_schema(raw, preset=preset, date_format=config.get("data.date_format"))
    sort_keys = [c for c in S.KEY_COLUMNS if c in mapped.columns]
    if sort_keys:
        mapped = mapped.sort_values(sort_keys).reset_index(drop=True)
    logger.info("Canonical columns: %s", list(mapped.columns))
    return mapped


def infer_preset(frame: pd.DataFrame) -> str:
    """Best-guess schema preset from the raw header."""
    cols = {c.lower() for c in frame.columns}
    if {"store_nbr", "family"} <= cols:
        return "favorita"
    if {"weekly_sales", "dept"} <= cols:
        return "walmart"
    if {"store", "promo"} <= cols and "sales" in cols:
        return "rossmann"
    return "canonical"
