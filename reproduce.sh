#!/usr/bin/env bash
# Reproduce the submission file on Linux or macOS.
#   ./reproduce.sh                 # data bundle unpacked into ./data
#   MTAH_DATA_DIR=/path ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
.venv/bin/python scripts/predict.py "$@"
