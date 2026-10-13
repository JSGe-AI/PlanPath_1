#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

GASTRIC_QWEN_HOST="${GASTRIC_QWEN_HOST:-127.0.0.1}"
GASTRIC_QWEN_PORT="${GASTRIC_QWEN_PORT:-8050}"
GASTRIC_QWEN_MODEL="${GASTRIC_QWEN_MODEL:-PlanPath-Qwen3.5-4B}"
GASTRIC_PATHO_HOST="${GASTRIC_PATHO_HOST:-127.0.0.1}"
GASTRIC_PATHO_PORT="${GASTRIC_PATHO_PORT:-8051}"
GASTRIC_PATHO_MODEL="${GASTRIC_PATHO_MODEL:-PlanPath-Patho-R1-7B}"
GASTRIC_CLIENT_GPU="${GASTRIC_CLIENT_GPU:-7}"

SHARD_COUNT="${1:-${SHARD_COUNT:-3}}"
RESULT_ROOT="${RESULT_ROOT:-outputs}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/full}"

require_port() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || ((10#${value} < 1 || 10#${value} > 65535)); then
    echo "${name} must be an integer from 1 to 65535." >&2
    exit 1
  fi
}

if (($# > 1)); then
  echo "Usage: bash scripts/launch_gastric_shards.sh [SHARD_COUNT]" >&2
  exit 1
fi
if [[ ! "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARD_COUNT must be a positive integer." >&2
  exit 1
fi
if [[ ! "${GASTRIC_CLIENT_GPU}" =~ ^[0-9]+$ ]]; then
  echo "GASTRIC_CLIENT_GPU must be one physical GPU index." >&2
  exit 1
fi
require_port "GASTRIC_QWEN_PORT" "${GASTRIC_QWEN_PORT}"
require_port "GASTRIC_PATHO_PORT" "${GASTRIC_PATHO_PORT}"
if [[ "${GASTRIC_QWEN_HOST}:${GASTRIC_QWEN_PORT}" == "${GASTRIC_PATHO_HOST}:${GASTRIC_PATHO_PORT}" ]]; then
  echo "Qwen and Patho-R1 must use different endpoints." >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
for endpoint in \
  "Qwen|http://${GASTRIC_QWEN_HOST}:${GASTRIC_QWEN_PORT}/v1/models" \
  "Patho-R1|http://${GASTRIC_PATHO_HOST}:${GASTRIC_PATHO_PORT}/v1/models"; do
  label="${endpoint%%|*}"
  url="${endpoint#*|}"
  if ! curl -fsS --max-time 5 "${url}" >/dev/null; then
    echo "${label} endpoint is unavailable: ${url}" >&2
    exit 1
  fi
done

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${RESULT_ROOT}/gastric/traces"
for ((shard_index=0; shard_index<SHARD_COUNT; shard_index++)); do
  output="${RESULT_ROOT}/gastric/predictions.shard_${shard_index}_of_${SHARD_COUNT}.jsonl"
  log="${LOG_DIR}/gastric.shard_${shard_index}_of_${SHARD_COUNT}.log"
  nohup env CUDA_VISIBLE_DEVICES="${GASTRIC_CLIENT_GPU}" \
    "${PYTHON_BIN}" -u scripts/run_planpath.py \
    --config configs/gastric.json \
    --device cuda:0 \
    --qwen-endpoint "http://${GASTRIC_QWEN_HOST}:${GASTRIC_QWEN_PORT}/v1" \
    --qwen-model "${GASTRIC_QWEN_MODEL}" \
    --patho-endpoint "http://${GASTRIC_PATHO_HOST}:${GASTRIC_PATHO_PORT}/v1" \
    --patho-model "${GASTRIC_PATHO_MODEL}" \
    --shard-unit patient \
    --shard-count "${SHARD_COUNT}" \
    --shard-index "${shard_index}" \
    --trace-root "${RESULT_ROOT}/gastric/traces" \
    --output "${output}" \
    >>"${log}" 2>&1 &
  echo "started gastric shard ${shard_index}/${SHARD_COUNT}: PID $!, log ${log}"
done
