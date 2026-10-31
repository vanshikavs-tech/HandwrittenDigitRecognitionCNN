#!/usr/bin/env bash
set -euo pipefail

# Public DHDD_CSV mirror of the Devanagari Handwritten Digits Dataset.
# 17,000 train and 3,000 test examples, 32x32 pixels per image.
# Dataset attribution: S. Acharya, A.K. Pant and P.K. Gyawali.
# CSV mirror: https://github.com/sknepal/DHDD_CSV

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/devanagari-csv"
mkdir -p "$DEST"

fetch() {
  local name="$1"
  local url="https://raw.githubusercontent.com/sknepal/DHDD_CSV/master/${name}"
  echo "Downloading ${name}..."
  curl --fail --location --retry 3 --output "$DEST/${name}.part" "$url"
  if [[ ! -s "$DEST/${name}.part" ]]; then
    echo "Download failed: ${name} is empty." >&2
    rm -f "$DEST/${name}.part"
    exit 1
  fi
  mv "$DEST/${name}.part" "$DEST/${name}"
}

fetch train.csv
fetch test.csv

echo
echo "Dataset installed in: $DEST"
echo "Training samples: 17,000; test samples: 3,000 (source dataset description)."
echo "Next, from the project root run:"
echo "  javac -d bin -sourcepath src src/Main.java test/NetworkTest.java"
echo "  java -cp bin Main --train-devanagari 5"
