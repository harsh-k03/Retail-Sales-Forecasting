"""Time-aware splitting. Random shuffling is never used anywhere in this project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.data import schema as S
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Fold:
    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> dict:
        return {
            "fold": self.index,
            "train_rows": int(len(self.train_idx)),
            "test_rows": int(len(self.test_idx)),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
        }


def walk_forward_splits(
    frame: pd.DataFrame,
    n_splits: int = 4,
    test_size_days: int = 30,
    gap_days: int = 0,
) -> List[Fold]:
    """Expanding-window folds: train on everything up to a cutoff, test the next block."""
    dates = frame[S.DATE]
    last = dates.max()
    folds: List[Fold] = []

    for i in range(n_splits):
        offset = (n_splits - 1 - i) * test_size_days
        test_end = last - pd.Timedelta(days=offset)
        test_start = test_end - pd.Timedelta(days=test_size_days - 1)
        train_end = test_start - pd.Timedelta(days=gap_days + 1)

        train_mask = dates <= train_end
        test_mask = (dates >= test_start) & (dates <= test_end)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.warning("Skipping empty fold %d", i)
            continue
        folds.append(
            Fold(
                index=i,
                train_idx=np.flatnonzero(train_mask.to_numpy()),
                test_idx=np.flatnonzero(test_mask.to_numpy()),
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def rolling_origin_splits(
    frame: pd.DataFrame,
    n_splits: int = 4,
    test_size_days: int = 30,
    train_window_days: int = 365,
    gap_days: int = 0,
) -> List[Fold]:
    """Sliding-window variant: fixed-length train window moves forward with the origin."""
    folds = walk_forward_splits(frame, n_splits, test_size_days, gap_days)
    dates = frame[S.DATE]
    out: List[Fold] = []
    for fold in folds:
        window_start = fold.train_end - pd.Timedelta(days=train_window_days)
        mask = (dates >= window_start) & (dates <= fold.train_end)
        out.append(
            Fold(fold.index, np.flatnonzero(mask.to_numpy()), fold.test_idx,
                 fold.train_end, fold.test_start, fold.test_end)
        )
    return out


def holdout_split(frame: pd.DataFrame, test_size_days: int = 30, gap_days: int = 0) -> Fold:
    """Single final split used for the reported test metrics and SHAP."""
    folds = walk_forward_splits(frame, n_splits=1, test_size_days=test_size_days, gap_days=gap_days)
    if not folds:
        raise ValueError("Not enough history for a holdout split")
    return folds[0]


def make_splits(frame: pd.DataFrame, config: Optional[Config] = None) -> List[Fold]:
    config = config or Config.load()
    cfg = config.get("split", {}) or {}
    strategy = cfg.get("strategy", "walk_forward")
    kwargs = dict(
        n_splits=int(cfg.get("n_splits", 4)),
        test_size_days=int(cfg.get("test_size_days", 30)),
        gap_days=int(cfg.get("gap_days", 0)),
    )
    if strategy == "rolling_origin":
        return rolling_origin_splits(frame, train_window_days=int(cfg.get("train_window_days", 365)), **kwargs)
    return walk_forward_splits(frame, **kwargs)


def iter_folds(frame: pd.DataFrame, config: Optional[Config] = None) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, Fold]]:
    for fold in make_splits(frame, config):
        yield frame.iloc[fold.train_idx], frame.iloc[fold.test_idx], fold


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Guard used in tests: every train date must precede every test date."""
    if train[S.DATE].max() >= test[S.DATE].min():
        raise AssertionError("Temporal leakage: train overlaps the test window")
