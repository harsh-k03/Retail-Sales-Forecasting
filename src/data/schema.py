"""Canonical schema definition shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

DATE = "date"
STORE = "store"
PRODUCT = "product"
TARGET = "sales"

REQUIRED_COLUMNS: List[str] = [DATE, STORE, PRODUCT, TARGET]

OPTIONAL_COLUMNS: List[str] = [
    "promotion",
    "holiday",
    "region",
    "category",
    "temperature",
    "inventory",
    "marketing_spend",
]

KEY_COLUMNS: List[str] = [DATE, STORE, PRODUCT]

# dtype family used by validation and the data dictionary
COLUMN_TYPES: Dict[str, str] = {
    DATE: "datetime",
    STORE: "categorical",
    PRODUCT: "categorical",
    TARGET: "numeric",
    "promotion": "binary",
    "holiday": "binary",
    "region": "categorical",
    "category": "categorical",
    "temperature": "numeric",
    "inventory": "numeric",
    "marketing_spend": "numeric",
}

DESCRIPTIONS: Dict[str, str] = {
    DATE: "Observation date of the sales record.",
    STORE: "Store / outlet identifier.",
    PRODUCT: "Product, SKU or product family identifier.",
    TARGET: "Units or revenue sold. Forecasting target.",
    "promotion": "1 if the item was on promotion that day, else 0.",
    "holiday": "1 if the date is a public/company holiday, else 0.",
    "region": "Geographic grouping of the store.",
    "category": "Product grouping / department.",
    "temperature": "Average daily temperature at the store location.",
    "inventory": "Stock on hand (or transactions proxy) for the day.",
    "marketing_spend": "Marketing or markdown spend attributed to the row.",
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    required: bool
    description: str


def column_specs() -> List[ColumnSpec]:
    specs = []
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        specs.append(
            ColumnSpec(
                name=col,
                dtype=COLUMN_TYPES[col],
                required=col in REQUIRED_COLUMNS,
                description=DESCRIPTIONS[col],
            )
        )
    return specs


def present_optional(columns) -> List[str]:
    return [c for c in OPTIONAL_COLUMNS if c in set(columns)]
