#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
RANKTABLE="${1:-ranktable_a3_2host_32rank.json}"
COUNT="${COUNT:-16}"
NP="${NP:-32}"
HOSTFILE="${HOSTFILE:-hostfile.mpi}"

export LD_LIBRARY_PATH="${ASCEND_HOME}/lib64:${LD_LIBRARY_PATH:-}"

MPI_ROOT_ARGS=()
if [ "$(id -u)" -eq 0 ] && mpirun --allow-run-as-root --version >/dev/null 2>&1; then
    MPI_ROOT_ARGS=(--allow-run-as-root)
fi

MPI_TAG_ARGS=()
if mpirun --tag-output --version >/dev/null 2>&1; then
    MPI_TAG_ARGS=(--tag-output)
fi

echo "[a3-smoke] host=$(hostname) pwd=$(pwd)"
echo "[a3-smoke] mpirun=$(command -v mpirun)"
mpirun --version | head -n 1 || true
echo "[a3-smoke] ranktable=${RANKTABLE} hostfile=${HOSTFILE} np=${NP} count=${COUNT}"
echo "[a3-smoke] launching mpirun"

mpirun "${MPI_ROOT_ARGS[@]}" "${MPI_TAG_ARGS[@]}" --hostfile "${HOSTFILE}" -np "${NP}" \
    ./hccl_allgather_smoke "${RANKTABLE}" "${COUNT}"
