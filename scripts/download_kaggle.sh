#!/usr/bin/env bash
# Download the Kaggle Store Sales competition data into data/raw/.
# Requires: pip install kaggle, and ~/.kaggle/kaggle.json with your API token.
set -euo pipefail

COMPETITION="${1:-store-sales-time-series-forecasting}"
TARGET="${2:-data/raw}"

mkdir -p "$TARGET"
kaggle competitions download -c "$COMPETITION" -p "$TARGET"
unzip -o "$TARGET/${COMPETITION}.zip" -d "$TARGET"
rm -f "$TARGET/${COMPETITION}.zip"

echo "Downloaded to $TARGET. Next: python scripts/prepare_kaggle.py"
