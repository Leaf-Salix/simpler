#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SIMPLER_REMOTE_L4_NPU_MACHINE_A:=120.9.10.37:19073}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_B:=120.9.10.35:19073}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICES:=0,1}"
: "${SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICES:=0,1}"
: "${SIMPLER_REMOTE_L4_NPU_PLATFORM:=a2a3}"
: "${SIMPLER_REMOTE_L4_NPU_RUNTIME:=tensormap_and_ringbuffer}"

cd "${ROOT_DIR}"
source .venv/bin/activate

exec python -m tools.remote_l4_npu.remote_l4_npu_smoke \
  --machine-a "${SIMPLER_REMOTE_L4_NPU_MACHINE_A}" \
  --machine-b "${SIMPLER_REMOTE_L4_NPU_MACHINE_B}" \
  --machine-a-devices "${SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICES}" \
  --machine-b-devices "${SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICES}" \
  --platform "${SIMPLER_REMOTE_L4_NPU_PLATFORM}" \
  --runtime "${SIMPLER_REMOTE_L4_NPU_RUNTIME}"
