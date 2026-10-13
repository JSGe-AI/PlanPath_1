#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-outputs}"
SHARD_COUNT="${1:-${SHARD_COUNT:-3}}"
DATASET="${2:-both}"

if (($# > 2)); then
  echo "Usage: bash scripts/merge_planpath_shards.sh [SHARD_COUNT] [renal|gastric|both]" >&2
  exit 1
fi
if [[ ! "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARD_COUNT must be a positive integer." >&2
  exit 1
fi
if [[ "${DATASET}" != "renal" && "${DATASET}" != "gastric" && "${DATASET}" != "both" ]]; then
  echo "Dataset must be renal, gastric, or both." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
datasets=("${DATASET}")
if [[ "${DATASET}" == "both" ]]; then
  datasets=(renal gastric)
fi
for cancer in "${datasets[@]}"; do
  inputs=()
  for ((shard_index=0; shard_index<SHARD_COUNT; shard_index++)); do
    shard="${RESULT_ROOT}/${cancer}/predictions.shard_${shard_index}_of_${SHARD_COUNT}.jsonl"
    if [[ ! -f "${shard}" ]]; then
      echo "Missing shard output: ${shard}" >&2
      exit 1
    fi
    inputs+=("${shard}")
  done
  expected_args=()
  if [[ -n "${EXPECTED_COUNT:-}" ]]; then
    expected_args=(--expected-count "${EXPECTED_COUNT}")
  fi
  "${PYTHON_BIN}" scripts/merge_planpath_shards.py \
    --output "${RESULT_ROOT}/${cancer}/predictions.jsonl" \
    "${expected_args[@]}" \
    "${inputs[@]}"
done
