"""Shared state for the API: model bundle, uploaded datasets, request history."""
from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from src.config import Config
from src.data.cleaning import clean
from src.data.loader import load_dataset
from src.data.validation import validate
from src.logger import get_logger
from src.models.persistence import ModelBundle, latest_bundle, load_bundle

logger = get_logger(__name__)


class ModelService:
    """Lazily loads the newest bundle and keeps uploaded datasets in memory."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self.started_at = time.time()
        self._bundle: Optional[ModelBundle] = None
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=int(self.config.get("api.history_size", 100)))

    # ------------------------------------------------------------ model
    @property
    def bundle(self) -> Optional[ModelBundle]:
        if self._bundle is None:
            path = latest_bundle(self.config.path("models"))
            if path:
                self._bundle = load_bundle(path)
                logger.info("Loaded model bundle %s", path)
        return self._bundle

    def reload_model(self) -> Optional[ModelBundle]:
        self._bundle = None
        return self.bundle

    @property
    def uptime(self) -> float:
        return time.time() - self.started_at

    # ---------------------------------------------------------- datasets
    def register_dataset(self, frame: pd.DataFrame) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        self._datasets[dataset_id] = frame
        return dataset_id

    def get_dataset(self, dataset_id: Optional[str]) -> pd.DataFrame:
        if dataset_id:
            if dataset_id not in self._datasets:
                raise KeyError(f"Unknown dataset_id '{dataset_id}'")
            return self._datasets[dataset_id]
        return self.default_dataset()

    def default_dataset(self) -> pd.DataFrame:
        if "__default__" not in self._datasets:
            source = self.config.get("data.sample_file")
            frame = load_dataset(source, self.config)
            cleaned, _ = clean(frame, self.config)
            self._datasets["__default__"] = cleaned
        return self._datasets["__default__"]

    def ingest_upload(self, source: Any, preset: Optional[str] = None):
        frame = load_dataset(source, self.config, preset=preset)
        report = validate(frame, self.config)
        cleaned, _ = clean(frame, self.config) if report.passed else (frame, {})
        dataset_id = self.register_dataset(cleaned)
        return dataset_id, frame, report

    # ----------------------------------------------------------- history
    def record(self, endpoint: str, **fields: Any) -> str:
        request_id = uuid.uuid4().hex[:12]
        entry = {
            "request_id": request_id,
            "endpoint": endpoint,
            "created_at": datetime.now(timezone.utc),
            **fields,
        }
        self._history.appendleft(entry)
        return request_id

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self._history)[:limit]


_service: Optional[ModelService] = None


def get_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
