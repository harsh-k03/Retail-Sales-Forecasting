"""Small shared helpers: seeding, timing, IO."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    logger.info("%s finished in %.2fs", label, time.perf_counter() - start)


def save_json(payload: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def memory_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(deep=True).sum() / 1024**2)


def downcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric dtypes in place-ish to keep large panels manageable."""
    out = frame.copy()
    for col in out.select_dtypes(include=["int64", "int32"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    return out
