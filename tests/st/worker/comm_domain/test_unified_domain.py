# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retained local VMM Buffer use across runs through the existing vector-add callable."""

from __future__ import annotations

import struct

import pytest
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from examples.workers.l3.multi_chip_dispatch.main import N_COLS, N_ROWS, build_chip_callable

_RUNTIME = "tensormap_and_ringbuffer"
_NUMEL = N_ROWS * N_COLS
_NBYTES = _NUMEL * 4


@pytest.mark.platforms(["a2a3sim", "a5sim", "a2a3", "a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_retained_cached_buffer_reaches_chip_in_second_run(st_platform, st_device_ids):
    worker = Worker(
        level=3, platform=st_platform, runtime=_RUNTIME, device_ids=[int(st_device_ids[0])], num_sub_workers=0
    )
    chip = worker.register(build_chip_callable(st_platform))
    request = dict(
        name="tp",
        members=[0],
        window_size=_NBYTES,
        buffers=[CommBufferSpec("payload", "float32", _NUMEL, _NBYTES)],
        retain_after_run=True,
    )
    saved = []
    try:
        worker.init()
        output = worker.create_buffer(_NBYTES)

        def first(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(**request)
            orch.copy_to_domain(domain, 0, struct.pack(f"<{_NUMEL}f", *([35.0] * _NUMEL)), buffer="payload")
            saved.append((domain, domain.local_context(0).buffers["payload"]))

        worker.run(first)
        domain, payload = saved[0]

        def second(orch, _args, cfg):
            # A cached VMM Buffer must enroll this run even without a domain lookup.
            args = TaskArgs()
            args.add_tensor(payload.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(payload.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(output.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(chip, args, cfg, worker=0)
            resources = worker._building_run_resources
            assert resources is not None and resources.live_domains.get("tp") is domain
            assert resources.requires_ordered_cleanup
            domain.release()
            assert domain.released and not domain.freed
            with pytest.raises(RuntimeError, match="released"):
                orch.submit_next_level(chip, args, cfg, worker=0)

        worker.run(second)
        assert domain.freed
        assert struct.unpack_from(f"<{_NUMEL}f", output.shm.buf) == (70.0,) * _NUMEL
    finally:
        worker.close()
