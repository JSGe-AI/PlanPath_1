#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${HISTOPLUS_MODEL_ROOT:-${PROJECT_ROOT}/models/histoplus}"

"${PYTHON_BIN}" -m pip install "${PROJECT_ROOT}/vendor/histoplus"
mkdir -p "${MODEL_ROOT}/hf_cache"

WEIGHT_20X="${MODEL_ROOT}/histoplus_cellvit_segmentor_20x.pt"
WEIGHT_40X="${MODEL_ROOT}/histoplus_cellvit_segmentor_40x.pt"
if [[ -s "${WEIGHT_20X}" && -s "${WEIGHT_40X}" ]]; then
  echo "Using existing local HistoPLUS checkpoints in ${MODEL_ROOT}"
else
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/download_histoplus_weights.py" \
    --model-root "${MODEL_ROOT}"
fi
