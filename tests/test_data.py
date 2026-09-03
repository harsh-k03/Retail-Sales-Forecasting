"""Loader, validation and cleaning behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import schema as S
from src.data.cleaning import clean, reindex_full_calendar
from src.data.loader import apply_schema, infer_preset, load_dataset
from src.data.validation import validate


def test_favorita_mapping_produces_canonical_columns(canonical_frame):
    assert set(S.REQUIRED_COLUMNS) <= set(canonical_frame.columns)
    assert pd.api.types.is_datetime64_any_dtype(canonical_frame[S.DATE])
    assert pd.api.types.is_numeric_dtype(canonical_frame[S.TARGET])


def test_promotion_coerced_to_binary(canonical_frame):
    assert set(canonical_frame["promotion"].unique()) <= {0, 1}


def test_constants_fill_missing_dimension():
    frame = pd.DataFrame({"Date": ["2022-01-01"], "Store": [1], "Sales": [10.0], "Promo": [1]})
    out = apply_schema(frame, preset="rossmann")
    assert out[S.PRODUCT].iloc[0] == "ALL"


def test_infer_preset_detects_favorita(raw_frame):
    assert infer_preset(raw_frame) == "favorita"


def test_validation_passes_on_clean_data(canonical_frame, config):
    assert validate(canonical_frame, config).passed


def test_validation_flags_missing_required_column(canonical_frame, config):
    report = validate(canonical_frame.drop(columns=[S.PRODUCT]), config)
    assert not report.passed
    assert report.errors[0].rule == "required_columns"


def test_validation_flags_duplicates(canonical_frame, config):
    duped = pd.concat([canonical_frame, canonical_frame.head(500)], ignore_index=True)
    report = validate(duped, config)
    assert any(i.rule == "duplicates" for i in report.issues)


def test_validation_flags_negative_sales(canonical_frame, config):
    frame = canonical_frame.copy()
    frame.loc[frame.index[:5], S.TARGET] = -1.0
    assert any(i.rule == "negative_sales" for i in validate(frame, config).issues)


def test_cleaning_removes_duplicates_and_negatives(canonical_frame, config):
    frame = pd.concat([canonical_frame, canonical_frame.head(100)], ignore_index=True)
    frame.loc[frame.index[:5], S.TARGET] = -20.0
    cleaned, stats = clean(frame, config)
    assert stats["dropped_duplicates"] >= 100
    assert (cleaned[S.TARGET] >= 0).all()
    assert not cleaned.duplicated(subset=S.KEY_COLUMNS).any()


def test_cleaning_is_idempotent(clean_frame, config):
    again, stats = clean(clean_frame, config)
    assert len(again) == len(clean_frame)
    assert stats["dropped_duplicates"] == 0


def test_reindex_full_calendar_has_no_gaps(clean_frame):
    subset = clean_frame[clean_frame[S.STORE] == clean_frame[S.STORE].iloc[0]]
    dense = reindex_full_calendar(subset)
    for _, group in dense.groupby([S.STORE, S.PRODUCT], observed=True):
        assert group[S.DATE].diff().dt.days.dropna().max() == 1


def test_missing_target_interpolation(canonical_frame, config):
    frame = canonical_frame.copy()
    frame.loc[frame.index[10:20], S.TARGET] = np.nan
    cleaned, _ = clean(frame, config)
    assert cleaned[S.TARGET].isna().sum() == 0


def test_load_dataset_accepts_dataframe(raw_frame, config):
    assert len(load_dataset(raw_frame, config, preset="favorita")) == len(raw_frame)


def test_unparseable_dates_are_dropped(canonical_frame, config):
    frame = canonical_frame.copy()
    frame.loc[frame.index[:3], S.DATE] = pd.NaT
    cleaned, stats = clean(frame, config)
    assert stats["dropped_invalid_dates"] == 3
    with pytest.raises(AssertionError):
        assert cleaned[S.DATE].isna().any()
