#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""No-mpirun A3 TLOAD smoke spanning one local L3 and one or more remote L3 nodes."""

from __future__ import annotations

import argparse
import struct
import sys

from global_tload_smoke import (
    COUNT,
    FLOAT_BYTES,
    MAX_RANKS,
    WINDOW_SIZE,
    _digest_scalars,
    _expected_values,
    _input_values,
    build_chip_callable,
)
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, Tensor, TensorArgType
from simpler.worker import CallableHandle, RemoteCallable, RemoteWorkerSpec, Worker


def _make_local_rank_orch(chip_handle: CallableHandle):
    def local_rank_orch(orch, args: TaskArgs, cfg: CallConfig) -> None:
        if args.scalar_count() != 2:
            raise ValueError("local TLOAD task expects domain_id and local_worker_id")
        domain_id = int(args.scalar(0))
        local_worker_id = int(args.scalar(1))
        context = orch.get_global_domain(domain_id)[local_worker_id]

        chip_args = TaskArgs()
        chip_args.add_tensor(
            Tensor.make(
                data=context.buffer_ptrs["input"],
                shapes=(COUNT,),
                dtype=DataType.FLOAT32,
                child_memory=True,
            ),
            TensorArgType.INPUT,
        )
        chip_args.add_tensor(
            Tensor.make(
                data=context.buffer_ptrs["result"],
                shapes=(COUNT,),
                dtype=DataType.FLOAT32,
                child_memory=True,
            ),
            TensorArgType.OUTPUT_EXISTING,
        )
        chip_args.add_scalar(context.domain_size)
        chip_args.add_scalar(context.device_ctx)
        orch.submit_next_level(chip_handle, chip_args, cfg, worker=local_worker_id)

    return local_rank_orch


def _domain_args(domain_id: int, local_worker_id: int) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(domain_id)
    args.add_scalar(local_worker_id)
    return args


def run(
    endpoints: list[str],
    remote_device_ids: list[int],
    local_device_id: int,
    platform: str,
    runtime: str,
) -> int:
    if len(endpoints) != len(remote_device_ids):
        raise ValueError("--endpoint and --remote-device-id counts must match")
    node_count = 1 + len(endpoints)
    if not 2 <= node_count <= MAX_RANKS:
        raise ValueError(f"the smoke requires between 2 and {MAX_RANKS} nodes including the local L3")

    chip_callable = build_chip_callable(platform, runtime)
    local_l3 = Worker(
        level=3,
        device_ids=[local_device_id],
        num_sub_workers=0,
        platform=platform,
        runtime=runtime,
        comm_profile="a3-fabric-v1",
        global_device_ranks=(0,),
    )
    local_chip_handle = local_l3.register(chip_callable)

    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=120)
    local_node_id = worker.add_worker(local_l3)
    remote_node_ids = tuple(
        worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=endpoints[index],
                platform=platform,
                runtime=runtime,
                device_ids=(remote_device_ids[index],),
                comm_profile="a3-fabric-v1",
                global_device_ranks=(index + 1,),
            )
        )
        for index in range(len(endpoints))
    )
    node_ids = (local_node_id, *remote_node_ids)

    print(f"[l4-mixed-global-tload] compiling for {platform}/{runtime}")
    remote_chip_handle = worker.register(chip_callable)
    local_orch_handle = worker.register(_make_local_rank_orch(local_chip_handle))
    remote_orch_handle = worker.register(
        RemoteCallable("simpler.global_comm_smoke:remote_rank_orch"),
        workers=list(remote_node_ids),
    )
    captured: dict[str, object] = {}
    try:
        worker.init()

        def build_and_run(orch, _args, cfg):
            domain = orch.allocate_global_domain(
                name="a3-l4-mixed-tload",
                members=tuple((node_id, 0) for node_id in node_ids),
                window_size=WINDOW_SIZE,
                buffers=(
                    CommBufferSpec("input", "float32", COUNT, COUNT * FLOAT_BYTES),
                    CommBufferSpec("result", "float32", COUNT, COUNT * FLOAT_BYTES),
                ),
                retain_after_run=True,
            )
            for rank in range(node_count):
                orch.copy_to_global_domain(
                    domain,
                    rank,
                    struct.pack(f"<{COUNT}f", *_input_values(rank)),
                    buffer="input",
                )

            orch.submit_next_level(
                local_orch_handle,
                _domain_args(domain.domain_id, 0),
                cfg,
                worker=local_node_id,
            )
            digest_scalars = _digest_scalars(remote_chip_handle.digest)
            for node_id in remote_node_ids:
                rank_args = _domain_args(domain.domain_id, 0)
                for value in digest_scalars:
                    rank_args.add_scalar(value)
                orch.submit_next_level(remote_orch_handle, rank_args, cfg, worker=node_id)
            captured["domain"] = domain

        worker.run(build_and_run, args=None, config=CallConfig())
        domain = captured["domain"]
        expected = _expected_values(node_count)
        observed: list[tuple[float, ...]] = []

        def read_and_release(orch, _args, _cfg):
            for rank in range(node_count):
                raw = orch.copy_from_global_domain(
                    domain,
                    rank,
                    COUNT * FLOAT_BYTES,
                    buffer="result",
                )
                observed.append(tuple(float(value) for value in struct.unpack(f"<{COUNT}f", raw)))
            domain.release()

        worker.run(read_and_release, args=None, config=CallConfig())
        for rank, result in enumerate(observed):
            max_diff = max(abs(actual - wanted) for actual, wanted in zip(result, expected, strict=True))
            print(f"[l4-mixed-global-tload] rank={rank} max_diff={max_diff:.3e}")
            if max_diff > 1e-3:
                print("[l4-mixed-global-tload] FAILED")
                return 1
        print("[l4-mixed-global-tload] PASS: local and remote L3 ranks completed peer TLOAD")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-device-id", required=True, type=int, help="Device owned by the local L3")
    parser.add_argument("--endpoint", action="append", required=True, help="Remote L3 daemon endpoint, HOST:PORT")
    parser.add_argument(
        "--remote-device-id",
        action="append",
        required=True,
        type=int,
        help="One device id per remote endpoint",
    )
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    args = parser.parse_args()
    return run(
        args.endpoint,
        args.remote_device_id,
        args.local_device_id,
        args.platform,
        args.runtime,
    )


if __name__ == "__main__":
    sys.exit(main())
