"""Shared fixtures. Everything runs on a small in-memory panel to stay fast."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from src.config import Config  # noqa: E402
from src.data.cleaning import clean  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.make_sample import generate  # noqa: E402
from src.features.builder import FeatureBuilder  # noqa: E402


@pytest.fixture(scope="session")
def config() -> Config:
    cfg = Config.load()
    cfg.data["split"]["n_splits"] = 2
    cfg.data["split"]["test_size_days"] = 14
    return cfg


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """Kaggle-schema raw frame: 2 stores x 2 families x ~1 year."""
    return generate(start="2022-01-01", end="2022-12-31", stores=[1, 3], families=["GROCERY I", "BEVERAGES"])


@pytest.fixture(scope="session")
def canonical_frame(raw_frame, config) -> pd.DataFrame:
    return load_dataset(raw_frame, config, preset="favorita")


@pytest.fixture(scope="session")
def clean_frame(canonical_frame, config) -> pd.DataFrame:
    cleaned, _ = clean(canonical_frame, config)
    return cleaned


@pytest.fixture(scope="session")
def builder(clean_frame, config) -> FeatureBuilder:
    b = FeatureBuilder(config)
    b.fit_transform(clean_frame)
    return b


@pytest.fixture(scope="session")
def feature_frame(clean_frame, config) -> pd.DataFrame:
    b = FeatureBuilder(config)
    return b.drop_warmup(b.fit_transform(clean_frame))


@pytest.fixture
def tmp_config(tmp_path, config) -> Config:
    cfg = Config.load()
    cfg.root = tmp_path
    cfg.data["split"] = dict(config.data["split"])
    return cfg
