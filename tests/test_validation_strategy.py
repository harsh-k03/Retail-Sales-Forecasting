"""Time-series splitting must never leak the future into training."""
from __future__ import annotations

import pytest

from src.data import schema as S
from src.validation.splitters import (
    assert_no_leakage,
    holdout_split,
    make_splits,
    rolling_origin_splits,
    walk_forward_splits,
)


def test_walk_forward_produces_requested_folds(feature_frame):
    folds = walk_forward_splits(feature_frame, n_splits=3, test_size_days=14)
    assert len(folds) == 3


def test_train_always_precedes_test(feature_frame):
    for fold in walk_forward_splits(feature_frame, n_splits=3, test_size_days=14):
        train = feature_frame.iloc[fold.train_idx]
        test = feature_frame.iloc[fold.test_idx]
        assert train[S.DATE].max() < test[S.DATE].min()
        assert_no_leakage(train, test)


def test_train_window_expands(feature_frame):
    folds = walk_forward_splits(feature_frame, n_splits=3, test_size_days=14)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)


def test_test_windows_do_not_overlap(feature_frame):
    folds = walk_forward_splits(feature_frame, n_splits=3, test_size_days=14)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end < later.test_start


def test_gap_is_respected(feature_frame):
    fold = walk_forward_splits(feature_frame, n_splits=1, test_size_days=14, gap_days=7)[0]
    assert (fold.test_start - fold.train_end).days == 8


def test_rolling_origin_keeps_a_bounded_window(feature_frame):
    folds = rolling_origin_splits(feature_frame, n_splits=2, test_size_days=14, train_window_days=90)
    for fold in folds:
        train = feature_frame.iloc[fold.train_idx]
        assert (train[S.DATE].max() - train[S.DATE].min()).days <= 91


def test_holdout_split_is_the_final_window(feature_frame):
    fold = holdout_split(feature_frame, test_size_days=14)
    assert fold.test_end == feature_frame[S.DATE].max()


def test_make_splits_uses_config(feature_frame, config):
    assert len(make_splits(feature_frame, config)) == config.get("split.n_splits")


def test_leakage_guard_raises(feature_frame):
    with pytest.raises(AssertionError):
        assert_no_leakage(feature_frame, feature_frame)
