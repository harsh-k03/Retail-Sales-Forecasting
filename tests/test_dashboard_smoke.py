"""Dashboard smoke tests: every page must import and expose the expected helpers."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "app" / "pages").glob("*.py"))
EXPECTED_PAGES = {
    "Upload", "Validation", "EDA", "Features", "Models",
    "Forecast", "Explainability", "Insights", "Scenario_Analysis", "Settings",
}


def test_every_blueprint_page_exists():
    names = {p.stem.split("_", 1)[1] for p in PAGES}
    assert EXPECTED_PAGES <= names


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.stem)
def test_pages_are_valid_python(path):
    ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.stem)
def test_pages_configure_the_page(path):
    assert "state.set_page(" in path.read_text(encoding="utf-8")


def test_home_page_parses():
    ast.parse((ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8"))


def test_components_import():
    from app.components import charts, state

    assert hasattr(state, "load_sample")
    assert hasattr(charts, "forecast_chart")
