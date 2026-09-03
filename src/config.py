"""Configuration loading and path resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_MODEL_PARAMS = PROJECT_ROOT / "config" / "model_params.yaml"
SCHEMA_DIR = PROJECT_ROOT / "config" / "schemas"


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    """Typed accessor over the YAML config tree."""

    data: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    @classmethod
    def load(
        cls,
        config_path: Optional[os.PathLike] = None,
        model_params_path: Optional[os.PathLike] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "Config":
        cfg = _read_yaml(Path(config_path or DEFAULT_CONFIG))
        params = _read_yaml(Path(model_params_path or DEFAULT_MODEL_PARAMS))
        if overrides:
            cfg = _deep_merge(cfg, overrides)
        return cls(data=cfg, model_params=params)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """Absolute path for a `paths.*` entry, created on demand."""
        rel = self.get(f"paths.{key}")
        if rel is None:
            raise KeyError(f"Unknown path key: {key}")
        resolved = self.root / rel
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def defaults_for(self, model_name: str) -> Dict[str, Any]:
        return dict(self.model_params.get("defaults", {}).get(model_name, {}))

    def search_space(self, model_name: str) -> Dict[str, Any]:
        return dict(self.model_params.get("search_spaces", {}).get(model_name, {}))

    @property
    def seed(self) -> int:
        return int(self.get("project.random_seed", 42))


def load_schema(preset: str) -> Dict[str, Any]:
    """Load a dataset schema preset from config/schemas."""
    path = SCHEMA_DIR / f"{preset}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in SCHEMA_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"Unknown schema preset '{preset}'. Available: {available}")
    return _read_yaml(path)


def available_schemas() -> list[str]:
    return sorted(p.stem for p in SCHEMA_DIR.glob("*.yaml"))


def load_config(**kwargs: Any) -> Config:
    return Config.load(**kwargs)
