#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${ROOT_DIR}/tools/remote_l4_npu/machine.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/tools/remote_l4_npu/machine.env"
  set +a
fi

: "${SIMPLER_REMOTE_L4_NPU_ROLE:=machine}"
: "${SIMPLER_REMOTE_L4_NPU_HOST:=0.0.0.0}"
: "${SIMPLER_REMOTE_L4_NPU_PORT:=19073}"
: "${SIMPLER_REMOTE_L4_NPU_DEVICE:=0}"

RUN_CMD="cd '${ROOT_DIR}'"
RUN_CMD+=" && source .venv/bin/activate"
RUN_CMD+=" && python -m simpler.remote_l3_worker"
RUN_CMD+=" --host '${SIMPLER_REMOTE_L4_NPU_HOST}'"
RUN_CMD+=" --port '${SIMPLER_REMOTE_L4_NPU_PORT}'"

echo "[remote-l4-npu] role=${SIMPLER_REMOTE_L4_NPU_ROLE}"
echo "[remote-l4-npu] listening on ${SIMPLER_REMOTE_L4_NPU_HOST}:${SIMPLER_REMOTE_L4_NPU_PORT}"
echo "[remote-l4-npu] device=${SIMPLER_REMOTE_L4_NPU_DEVICE}"

exec bash -lc "${RUN_CMD}"
