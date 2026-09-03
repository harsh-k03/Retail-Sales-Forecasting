"""Assemble the modelling matrix from the cleaned canonical frame."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.data import schema as S
from src.features.calendar import (
    add_calendar_features,
    add_event_features,
    add_fourier_terms,
    calendar_feature_names,
)
from src.features.encoders import OrdinalEncoder
from src.features.lags import (
    add_diff_features,
    add_expanding_features,
    add_lag_features,
    add_rolling_features,
    lag_feature_names,
    max_lag,
)
from src.logger import get_logger

logger = get_logger(__name__)

CATEGORICAL_SOURCES = [S.STORE, S.PRODUCT, "region", "category"]
EXOGENOUS = ["promotion", "holiday", "temperature", "inventory", "marketing_spend"]


@dataclass
class FeatureSpec:
    """Everything needed to rebuild the same matrix at inference time."""

    feature_names: List[str] = field(default_factory=list)
    categorical_codes: List[str] = field(default_factory=list)
    target: str = S.TARGET
    lags: List[int] = field(default_factory=list)
    rolling_windows: List[int] = field(default_factory=list)
    warmup_rows: int = 0
    target_transform: str = "none"


class FeatureBuilder:
    """Stateful builder: fit encoders once, reuse for train/valid/forecast."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        f = self.config.get("features", {}) or {}
        self.lags = list(f.get("lags", [1, 7, 14, 28]))
        self.windows = list(f.get("rolling_windows", [7, 28]))
        self.stats = list(f.get("rolling_stats", ["mean", "std"]))
        self.use_calendar = bool(f.get("add_calendar", True))
        self.use_fourier = bool(f.get("add_fourier", True))
        self.fourier_periods = list(f.get("fourier_periods", [7, 365.25]))
        self.fourier_order = int(f.get("fourier_order", 3))
        self.use_interactions = bool(f.get("add_promo_interactions", True))
        self.target_transform = f.get("target_transform", "none")
        self.encoder = OrdinalEncoder(columns=list(CATEGORICAL_SOURCES))
        self.spec = FeatureSpec(
            lags=self.lags,
            rolling_windows=self.windows,
            warmup_rows=max_lag(self.lags, self.windows),
            target_transform=self.target_transform,
        )

    # ------------------------------------------------------------------ build
    def _engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.sort_values(S.KEY_COLUMNS).reset_index(drop=True)
        if self.use_calendar:
            out = add_calendar_features(out)
        if self.use_fourier:
            out = add_fourier_terms(out, self.fourier_periods, self.fourier_order)
        out = add_event_features(out)
        out = add_lag_features(out, self.lags)
        out = add_rolling_features(out, self.windows, self.stats)
        out = add_expanding_features(out)
        out = add_diff_features(out, [1, 7])
        if self.use_interactions:
            out = self._interactions(out)
        return out

    @staticmethod
    def _interactions(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        lag7 = f"{S.TARGET}_roll7_mean"
        if "promotion" in out.columns and lag7 in out.columns:
            out["promo_x_recent_mean"] = out["promotion"] * out[lag7]
        if "holiday" in out.columns and "is_weekend" in out.columns:
            out["holiday_x_weekend"] = out["holiday"] * out["is_weekend"]
        if "promotion" in out.columns and "dayofweek" in out.columns:
            out["promo_x_dow"] = out["promotion"] * (out["dayofweek"] + 1)
        if "marketing_spend" in out.columns and "promotion" in out.columns:
            out["spend_x_promo"] = out["marketing_spend"] * out["promotion"]
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self._engineer(frame)
        out = self.encoder.fit_transform(out)
        self.spec.categorical_codes = self.encoder.code_columns
        self.spec.feature_names = self._collect_feature_names(out)
        logger.info("Built %d features from %d rows", len(self.spec.feature_names), len(out))
        return out

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self._engineer(frame)
        out = self.encoder.transform(out)
        for col in self.spec.feature_names:
            if col not in out.columns:
                out[col] = np.nan
        return out

    def _collect_feature_names(self, frame: pd.DataFrame) -> List[str]:
        names: List[str] = []
        names += calendar_feature_names(frame)
        names += [c for c in frame.columns if c.startswith(("sin_", "cos_"))]
        names += [c for c in EXOGENOUS if c in frame.columns]
        names += [c for c in frame.columns if c.endswith(("_lead_1", "_lag_1", "_roll_7")) and c.startswith(("holiday", "promotion"))]
        names += lag_feature_names(frame)
        names += self.encoder.code_columns
        names += [c for c in ("promo_x_recent_mean", "holiday_x_weekend", "promo_x_dow", "spend_x_promo") if c in frame.columns]
        seen, ordered = set(), []
        for name in names:
            if name not in seen and name in frame.columns:
                seen.add(name)
                ordered.append(name)
        return ordered

    # ------------------------------------------------------------------- misc
    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.spec.feature_names]

    def drop_warmup(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove rows whose lag features are still undefined."""
        required = [c for c in self.spec.feature_names if c.startswith(f"{S.TARGET}_lag_")]
        if not required:
            return frame
        return frame.dropna(subset=required).reset_index(drop=True)


def build_features(frame: pd.DataFrame, config: Optional[Config] = None):
    """Convenience wrapper returning (features_frame, builder)."""
    builder = FeatureBuilder(config)
    return builder.fit_transform(frame), builder
