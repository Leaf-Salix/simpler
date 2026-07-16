#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SIMPLER_REMOTE_L4_NPU_MACHINE_A:?set to host:port for machineA, for example 10.0.0.11:19073}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_B:?set to host:port for machineB, for example 10.0.0.12:19073}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICE:=0}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICE:=0}"
: "${SIMPLER_REMOTE_L4_NPU_PLATFORM:=a2a3}"
: "${SIMPLER_REMOTE_L4_NPU_RUNTIME:=tensormap_and_ringbuffer}"

cd "${ROOT_DIR}"
source .venv/bin/activate

exec python -m tools.remote_l4_npu.remote_l4_npu_smoke \
  --machine-a "${SIMPLER_REMOTE_L4_NPU_MACHINE_A}" \
  --machine-b "${SIMPLER_REMOTE_L4_NPU_MACHINE_B}" \
  --machine-a-device "${SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICE}" \
  --machine-b-device "${SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICE}" \
  --platform "${SIMPLER_REMOTE_L4_NPU_PLATFORM}" \
  --runtime "${SIMPLER_REMOTE_L4_NPU_RUNTIME}"
