"""What-if simulator: re-forecast under modified drivers and compare."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import Config
from src.data import schema as S
from src.features.builder import FeatureBuilder
from src.forecast.recursive import forecast
from src.logger import get_logger
from src.models.base import BaseForecaster

logger = get_logger(__name__)

SUPPORTED_LEVERS = ["promotion", "holiday", "marketing_spend", "inventory", "temperature"]


@dataclass
class Scenario:
    """A named set of driver overrides applied to the forecast window."""

    name: str = "scenario"
    overrides: Dict[str, Any] = field(default_factory=dict)
    horizon: int = 30

    def describe(self) -> str:
        if not self.overrides:
            return "baseline"
        return ", ".join(f"{k}={v}" for k, v in self.overrides.items())


class ScenarioSimulator:
    """Runs a baseline forecast once and re-runs it per scenario."""

    def __init__(
        self,
        model: BaseForecaster,
        history: pd.DataFrame,
        builder: FeatureBuilder,
        config: Optional[Config] = None,
    ) -> None:
        self.model = model
        self.history = history
        self.builder = builder
        self.config = config or Config.load()
        self._baseline: Dict[int, pd.DataFrame] = {}

    def baseline(self, horizon: int = 30) -> pd.DataFrame:
        if horizon not in self._baseline:
            self._baseline[horizon] = forecast(
                self.model, self.history, self.builder, horizon=horizon, config=self.config
            )
        return self._baseline[horizon]

    def run(self, scenario: Scenario) -> pd.DataFrame:
        overrides = {k: v for k, v in scenario.overrides.items() if k in SUPPORTED_LEVERS}
        unknown = set(scenario.overrides) - set(overrides)
        if unknown:
            logger.warning("Ignoring unsupported levers: %s", sorted(unknown))
        return forecast(
            self.model, self.history, self.builder,
            horizon=scenario.horizon, overrides=overrides, config=self.config,
        )

    def compare(self, scenario: Scenario) -> Dict[str, Any]:
        base = self.baseline(scenario.horizon)
        variant = self.run(scenario)
        base_total = float(base["forecast"].sum())
        variant_total = float(variant["forecast"].sum())
        delta = variant_total - base_total
        return {
            "scenario": scenario.name,
            "levers": scenario.describe(),
            "horizon_days": scenario.horizon,
            "baseline_total": base_total,
            "scenario_total": variant_total,
            "delta": delta,
            "delta_pct": (delta / base_total * 100) if base_total else 0.0,
            "baseline_frame": base,
            "scenario_frame": variant,
        }

    def sweep(self, lever: str, values: List[Any], horizon: int = 30) -> pd.DataFrame:
        """Sensitivity curve for one lever."""
        rows = []
        for value in values:
            result = self.compare(Scenario(f"{lever}={value}", {lever: value}, horizon))
            rows.append({
                "lever": lever,
                "value": value,
                "total_forecast": result["scenario_total"],
                "delta_pct": result["delta_pct"],
            })
        return pd.DataFrame(rows)

    def daily_comparison(self, scenario: Scenario) -> pd.DataFrame:
        result = self.compare(scenario)
        base = result["baseline_frame"].groupby(S.DATE, observed=True)["forecast"].sum().rename("baseline")
        variant = result["scenario_frame"].groupby(S.DATE, observed=True)["forecast"].sum().rename(scenario.name)
        out = pd.concat([base, variant], axis=1).reset_index()
        out["delta"] = out[scenario.name] - out["baseline"]
        return out


def promo_scenarios(horizon: int = 30) -> List[Scenario]:
    """Standard set used by the report and the dashboard defaults."""
    return [
        Scenario("no_promotion", {"promotion": 0}, horizon),
        Scenario("always_promotion", {"promotion": 1}, horizon),
        Scenario("holiday_period", {"holiday": 1}, horizon),
    ]
