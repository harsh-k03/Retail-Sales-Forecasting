#!/usr/bin/env python3
"""Regenerate the demo CSV used by tests, CI and the dashboard's example mode."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.make_sample import write  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the sample Kaggle-schema dataset")
    parser.add_argument("--out", default="data/sample/sample_store_sales.csv")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(write(Path(args.out), start=args.start, end=args.end, seed=args.seed))
