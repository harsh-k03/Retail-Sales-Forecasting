"""Generate a small demo file that mirrors the Kaggle Store Sales raw schema.

Real data goes in data/raw/ (see scripts/download_kaggle.sh). This sample exists
so the pipeline, tests and CI can run without downloading the competition data.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

STORES = [1, 2, 3, 5, 8]
FAMILIES = ["GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING"]
STATES = {1: "Pichincha", 2: "Pichincha", 3: "Guayas", 5: "Azuay", 8: "Guayas"}
STORE_TYPES = {1: "D", 2: "D", 3: "A", 5: "B", 8: "C"}
FAMILY_BASE = {"GROCERY I": 480.0, "BEVERAGES": 310.0, "PRODUCE": 220.0, "CLEANING": 130.0}
STORE_SCALE = {1: 1.0, 2: 0.85, 3: 1.35, 5: 0.6, 8: 1.1}


def _holidays(index: pd.DatetimeIndex) -> pd.Series:
    """Fixed-date national holidays plus the late-December peak."""
    md = index.strftime("%m-%d")
    fixed = {"01-01", "05-01", "08-10", "11-02", "11-03", "12-25", "12-31"}
    return pd.Series(np.isin(md, list(fixed)).astype("int8"), index=index)


def generate(
    start: str = "2021-01-01",
    end: str = "2022-12-31",
    stores: Optional[List[int]] = None,
    families: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    stores = stores or STORES
    families = families or FAMILIES

    holiday_flag = _holidays(dates)
    day_of_year = dates.dayofyear.to_numpy()
    yearly = 1 + 0.18 * np.sin(2 * np.pi * (day_of_year - 20) / 365.25)
    weekly = np.array([0.92, 0.88, 0.9, 0.95, 1.12, 1.35, 1.18])[dates.dayofweek.to_numpy()]
    trend = np.linspace(1.0, 1.22, len(dates))
    oil = 55 + np.cumsum(rng.normal(0, 0.45, len(dates)))
    temperature = 21 + 4 * np.sin(2 * np.pi * (day_of_year - 30) / 365.25) + rng.normal(0, 1.4, len(dates))

    rows = []
    for store in stores:
        store_noise = 1 + rng.normal(0, 0.04)
        transactions = np.maximum(
            120, (900 * STORE_SCALE[store] * weekly * trend + rng.normal(0, 45, len(dates))).round()
        ).astype(int)
        for family in families:
            promo = rng.random(len(dates)) < 0.14
            onpromotion = (promo * rng.integers(1, 25, len(dates))).astype(int)
            base = FAMILY_BASE[family] * STORE_SCALE[store] * store_noise
            level = base * yearly * weekly * trend
            level = level * (1 + 0.22 * promo) * (1 + 0.30 * holiday_flag.to_numpy())
            level = level * (1 - 0.004 * (temperature - temperature.mean()))
            noise = rng.normal(1.0, 0.09, len(dates))
            sales = np.maximum(0.0, level * noise).round(3)
            # a few closed days per series
            closed = rng.random(len(dates)) < 0.004
            sales[closed] = 0.0

            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store_nbr": store,
                        "family": family,
                        "sales": sales,
                        "onpromotion": onpromotion,
                        "state": STATES[store],
                        "type": STORE_TYPES[store],
                        "cluster": (store % 4) + 1,
                        "transactions": transactions,
                        "dcoilwtico": oil.round(2),
                        "temperature": temperature.round(2),
                        "is_holiday": holiday_flag.to_numpy(),
                    }
                )
            )

    frame = pd.concat(rows, ignore_index=True)
    frame.insert(0, "id", np.arange(len(frame)))
    return frame.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)


def write(path: Path = Path("data/sample/sample_store_sales.csv"), **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(**kwargs).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    print(write())
