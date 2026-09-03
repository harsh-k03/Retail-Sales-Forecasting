#!/usr/bin/env python3
"""Join the Kaggle Store Sales files into one table the favorita preset can read.

Expects the competition files in data/raw/:
  train.csv, stores.csv, transactions.csv, oil.csv, holidays_events.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logger import get_logger  # noqa: E402

logger = get_logger("prepare_kaggle")


def prepare(raw_dir: Path, out_path: Path, max_rows: int | None = None) -> Path:
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["date"])
    if max_rows:
        train = train.tail(max_rows)
    logger.info("train.csv: %s", train.shape)

    stores_path = raw_dir / "stores.csv"
    if stores_path.exists():
        stores = pd.read_csv(stores_path)
        train = train.merge(stores, on="store_nbr", how="left")

    tx_path = raw_dir / "transactions.csv"
    if tx_path.exists():
        tx = pd.read_csv(tx_path, parse_dates=["date"])
        train = train.merge(tx, on=["date", "store_nbr"], how="left")

    oil_path = raw_dir / "oil.csv"
    if oil_path.exists():
        oil = pd.read_csv(oil_path, parse_dates=["date"])
        oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
        train = train.merge(oil, on="date", how="left")

    hol_path = raw_dir / "holidays_events.csv"
    if hol_path.exists():
        holidays = pd.read_csv(hol_path, parse_dates=["date"])
        national = holidays[(holidays["locale"] == "National") & (~holidays["transferred"])]
        flags = national[["date"]].drop_duplicates().assign(is_holiday=1)
        train = train.merge(flags, on="date", how="left")
        train["is_holiday"] = train["is_holiday"].fillna(0).astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_path, index=False)
    logger.info("Wrote %s (%s rows)", out_path, len(train))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join Kaggle Store Sales files")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/interim/store_sales_joined.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Keep only the most recent N rows")
    args = parser.parse_args()
    prepare(Path(args.raw_dir), Path(args.out), args.max_rows)
