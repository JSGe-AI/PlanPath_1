#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RENAL_QWEN_HOST="${RENAL_QWEN_HOST:-127.0.0.1}"
RENAL_QWEN_PORT="${RENAL_QWEN_PORT:-8050}"
RENAL_QWEN_MODEL="${RENAL_QWEN_MODEL:-PlanPath-Qwen3.5-4B}"
RENAL_PATHO_HOST="${RENAL_PATHO_HOST:-127.0.0.1}"
RENAL_PATHO_PORT="${RENAL_PATHO_PORT:-8051}"
RENAL_PATHO_MODEL="${RENAL_PATHO_MODEL:-PlanPath-Patho-R1-7B}"
RENAL_CLIENT_GPU="${RENAL_CLIENT_GPU:-0}"

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
  echo "Usage: bash scripts/launch_renal_shards.sh [SHARD_COUNT]" >&2
  exit 1
fi
if [[ ! "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARD_COUNT must be a positive integer." >&2
  exit 1
fi
if [[ ! "${RENAL_CLIENT_GPU}" =~ ^[0-9]+$ ]]; then
  echo "RENAL_CLIENT_GPU must be one physical GPU index." >&2
  exit 1
fi
require_port "RENAL_QWEN_PORT" "${RENAL_QWEN_PORT}"
require_port "RENAL_PATHO_PORT" "${RENAL_PATHO_PORT}"
if [[ "${RENAL_QWEN_HOST}:${RENAL_QWEN_PORT}" == "${RENAL_PATHO_HOST}:${RENAL_PATHO_PORT}" ]]; then
  echo "Qwen and Patho-R1 must use different endpoints." >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
for endpoint in \
  "Qwen|http://${RENAL_QWEN_HOST}:${RENAL_QWEN_PORT}/v1/models" \
  "Patho-R1|http://${RENAL_PATHO_HOST}:${RENAL_PATHO_PORT}/v1/models"; do
  label="${endpoint%%|*}"
  url="${endpoint#*|}"
  if ! curl -fsS --max-time 5 "${url}" >/dev/null; then
    echo "${label} endpoint is unavailable: ${url}" >&2
    exit 1
  fi
done

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${RESULT_ROOT}/renal/traces"
for ((shard_index=0; shard_index<SHARD_COUNT; shard_index++)); do
  output="${RESULT_ROOT}/renal/predictions.shard_${shard_index}_of_${SHARD_COUNT}.jsonl"
  log="${LOG_DIR}/renal.shard_${shard_index}_of_${SHARD_COUNT}.log"
  nohup env CUDA_VISIBLE_DEVICES="${RENAL_CLIENT_GPU}" \
    "${PYTHON_BIN}" -u scripts/run_planpath.py \
    --config configs/renal.json \
    --device cuda:0 \
    --qwen-endpoint "http://${RENAL_QWEN_HOST}:${RENAL_QWEN_PORT}/v1" \
    --qwen-model "${RENAL_QWEN_MODEL}" \
    --patho-endpoint "http://${RENAL_PATHO_HOST}:${RENAL_PATHO_PORT}/v1" \
    --patho-model "${RENAL_PATHO_MODEL}" \
    --shard-unit patient \
    --shard-count "${SHARD_COUNT}" \
    --shard-index "${shard_index}" \
    --trace-root "${RESULT_ROOT}/renal/traces" \
    --output "${output}" \
    >>"${log}" 2>&1 &
  echo "started renal shard ${shard_index}/${SHARD_COUNT}: PID $!, log ${log}"
done
