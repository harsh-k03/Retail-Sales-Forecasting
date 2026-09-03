"""Configuration and schema preset loading."""
from __future__ import annotations

import pytest

from src.config import Config, available_schemas, load_schema


def test_config_loads_expected_sections(config: Config):
    for section in ("project", "paths", "data", "validation", "features", "split", "training", "forecast"):
        assert config.get(section) is not None


def test_dotted_lookup_and_default(config: Config):
    assert config.get("training.target") == "sales"
    assert config.get("does.not.exist", "fallback") == "fallback"


def test_path_helper_creates_directory(tmp_config):
    path = tmp_config.path("figures")
    assert path.exists() and path.is_dir()


def test_model_defaults_and_search_space(config: Config):
    assert config.defaults_for("lightgbm")["n_estimators"] > 0
    assert "learning_rate" in config.search_space("lightgbm")


def test_every_preset_declares_required_mappings():
    for preset in available_schemas():
        spec = load_schema(preset)
        mapped = set(spec.get("columns", {})) | set(spec.get("constants", {}))
        assert {"date", "store", "sales"} <= mapped, preset


def test_unknown_preset_raises():
    with pytest.raises(FileNotFoundError):
        load_schema("not_a_preset")
