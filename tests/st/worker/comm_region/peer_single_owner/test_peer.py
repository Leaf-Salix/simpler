# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two device processes use the same provider payload and counter backing."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler

_RUNTIME = "tensormap_and_ringbuffer"
_WORDS = 128
_BYTES = _WORDS * 4


def _callable(platform):
    compiler = KernelCompiler(platform=platform)
    binary = compiler.compile_orchestration(
        runtime_name=_RUNTIME,
        source_path=str(Path(__file__).parent / "kernels/orchestration/peer_orch.cpp"),
    )
    return ChipCallable.build(signature=[D.OUT], func_name="peer_orchestration", binary=binary, children=[])


def _task_args(binding, output, phase, seed):
    args = TaskArgs()
    args.add_tensor(output.tensor([_WORDS], DataType.UINT32), TensorArgType.OUTPUT_EXISTING)
    for value in (*binding, phase, seed):
        args.add_scalar(int(value))
    return args


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(_RUNTIME)
def test_peer_same_backing_two_lifecycles(st_platform, st_device_ids):
    worker = Worker(
        level=3,
        device_ids=[int(value) for value in st_device_ids[:2]],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    chip = worker.register(_callable(st_platform))
    saved = []
    saved_bindings = []
    try:
        worker.init()
        outputs = [worker.create_buffer(_BYTES) for _ in range(3)]
        for seed in (17, 513):

            def orch(handle, _args, cfg):
                if saved_bindings:
                    with pytest.raises(ValueError, match="expired|submitted chip"):
                        handle.submit_next_level(
                            chip,
                            _task_args(saved_bindings[-1], outputs[1], 1, seed),
                            cfg,
                            worker=1,
                        )
                region = handle.create_worker_chip_region(
                    worker_id=0,
                    consumer_worker_id=1,
                    payload_bytes=_BYTES,
                    counter_bytes=128,
                )
                saved.append(region)
                owner = region.descriptor_scalars(worker_id=0)
                peer = region.descriptor_scalars()
                assert owner[1] == peer[1]
                assert tuple(owner[3::2]) == tuple(peer[3::2]) == (_BYTES, 128)
                saved_bindings.append(peer)
                if owner != peer:
                    with pytest.raises(ValueError, match="binding|submitted chip"):
                        handle.submit_next_level(chip, _task_args(peer, outputs[1], 1, seed), cfg, worker=0)
                altered = list(peer)
                altered[2] += 64
                with pytest.raises(ValueError, match="binding|submitted chip"):
                    handle.submit_next_level(chip, _task_args(altered, outputs[1], 1, seed), cfg, worker=1)
                previous = None
                for phase, target in enumerate((0, 1, 0)):
                    args = _task_args(owner if target == 0 else peer, outputs[phase], phase, seed)
                    if previous is not None:
                        args.add_dep(previous)
                    previous = handle.submit_next_level(chip, args, cfg, worker=target)

            worker.run(orch)
            initial = tuple(seed + i for i in range(_WORDS))
            changed = tuple(value ^ 0xFFFF for value in initial)
            for output, expected in zip(outputs, (initial, changed, changed)):
                assert struct.unpack_from(f"<{_WORDS}I", output.shm.buf) == expected
            assert worker._region_instance_registry._instances == {}
            with pytest.raises(RuntimeError, match="expired|released"):
                saved[-1].descriptor_scalars()
    finally:
        worker.close()
