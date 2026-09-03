"""Categorical encoding that is fitted on train only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

UNKNOWN = -1


@dataclass
class OrdinalEncoder:
    """Deterministic label encoding with an explicit unknown bucket."""

    columns: List[str] = field(default_factory=list)
    mapping: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "OrdinalEncoder":
        self.mapping = {}
        for col in self.columns:
            if col not in frame.columns:
                continue
            categories = sorted(frame[col].dropna().astype(str).unique())
            self.mapping[col] = {value: idx for idx, value in enumerate(categories)}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col, table in self.mapping.items():
            if col in out.columns:
                out[f"{col}_code"] = out[col].astype(str).map(table).fillna(UNKNOWN).astype("int32")
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    @property
    def code_columns(self) -> List[str]:
        return [f"{col}_code" for col in self.mapping]
