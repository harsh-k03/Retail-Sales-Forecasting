"""Serialize a trained model together with everything needed to reuse it."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.logger import get_logger

logger = get_logger(__name__)
BUNDLE_VERSION = 1


@dataclass
class ModelBundle:
    """Model + feature contract + provenance, stored as a single joblib file."""

    model: Any
    builder: Any
    feature_names: List[str]
    target: str = "sales"
    model_name: str = "unknown"
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = BUNDLE_VERSION

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target": self.target,
            "n_features": len(self.feature_names),
            "metrics": self.metrics,
            "created_at": self.created_at,
            "version": self.version,
            **{k: v for k, v in self.metadata.items() if not isinstance(v, (list, dict))},
        }


def save_bundle(bundle: ModelBundle, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved model bundle -> %s", path)
    return path


def load_bundle(path: Path) -> ModelBundle:
    bundle = joblib.load(Path(path))
    if getattr(bundle, "version", 0) != BUNDLE_VERSION:
        logger.warning("Bundle version mismatch: %s", getattr(bundle, "version", None))
    return bundle


def latest_bundle(models_dir: Path, pattern: str = "*.joblib") -> Optional[Path]:
    files = sorted(Path(models_dir).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def bundle_path(models_dir: Path, model_name: str) -> Path:
    return Path(models_dir) / f"{model_name}.joblib"


def write_metadata(bundle: ModelBundle, path: Path) -> Path:
    from src.utils import save_json

    payload = bundle.summary()
    payload["feature_names"] = bundle.feature_names
    return save_json(payload, Path(path))
