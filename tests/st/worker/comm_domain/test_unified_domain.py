# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Domain reuse and existing node-local Tensor dispatch on real workers."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import (
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_NUMEL = 128 * 128
_NBYTES = _NUMEL * 4


def _chip_callable(platform):
    compiler = KernelCompiler(platform=platform)
    source = Path(__file__).parent / "kernels/orchestration/domain_tensor_orch.cpp"
    binary = compiler.compile_orchestration(runtime_name=_RUNTIME, source_path=str(source))
    incore_source = Path(__file__).parents[1] / "comm_region/recursive_single_owner/kernels/aiv/kernel_transform.cpp"
    incore = compiler.compile_incore(
        str(incore_source),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(_RUNTIME),
    )
    if not platform.endswith("sim"):
        incore = extract_text_section(incore)
    return ChipCallable.build(
        signature=[D.IN, D.OUT],
        func_name="domain_tensor_orch",
        binary=binary,
        children=[(0, CoreCallable.build(signature=[D.IN, D.OUT, D.SCALAR], binary=incore))],
    )


def _worker(platform, device_ids, **config):
    return Worker(level=3, platform=platform, runtime=_RUNTIME, device_ids=device_ids, num_sub_workers=0, **config)


@pytest.mark.platforms(["a2a3sim", "a5sim", "a2a3", "a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_retained_cached_buffer_reaches_chip_in_second_run(st_platform, st_device_ids):
    worker = _worker(st_platform, [int(st_device_ids[0])])
    chip = worker.register(_chip_callable(st_platform))
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
            args.add_tensor(output.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(chip, args, cfg, worker=0)
            domain.release()
            assert domain.released and not domain.freed

        worker.run(second)
        assert domain.freed
        assert struct.unpack_from(f"<{_NUMEL}f", output.shm.buf) == (42.0,) * _NUMEL
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_l4_reuse_keeps_existing_l3_local_view_dispatch(st_platform, st_device_ids):
    child = _worker(st_platform, [int(st_device_ids[0])], comm_profile="sim")
    parent = Worker(level=4, num_sub_workers=0)
    callable_obj = _chip_callable(st_platform)
    chip = child.register(callable_obj)
    node = parent.add_worker(child)
    parent.register(callable_obj)

    def forward(orch, args, cfg):
        view = orch.get_global_domain(args.scalar(0))
        local = view[0]
        chip_args = TaskArgs()
        chip_args.add_tensor(local.buffers["payload"].tensor([_NUMEL], DataType.FLOAT32), TensorArgType.INPUT)
        chip_args.add_tensor(args.tensor(0), TensorArgType.OUTPUT_EXISTING)
        orch.submit_next_level(chip, chip_args, cfg, worker=0)

    forward_handle = parent.register(forward)
    request = dict(
        name="tp",
        members=[(node, 0)],
        window_size=_NBYTES,
        buffers=[CommBufferSpec("payload", "float32", _NUMEL, _NBYTES)],
        retain_after_run=True,
    )
    saved = []
    try:
        parent.init()
        output = parent.create_buffer(_NBYTES)

        def first(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(**request)
            orch.copy_to_domain(domain, 0, struct.pack(f"<{_NUMEL}f", *([35.0] * _NUMEL)), buffer="payload")
            saved.append(domain)

        parent.run(first)

        def second(orch, _args, cfg):
            domain = orch.get_or_allocate_domain(**request)
            assert domain is saved[0]
            args = TaskArgs()
            args.add_tensor(output.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            args.add_scalar(domain.domain_id)
            orch.submit_next_level(forward_handle, args, cfg, worker=node)
            domain.release()
            assert not domain.freed

        parent.run(second)
        assert saved[0].freed
        assert struct.unpack_from(f"<{_NUMEL}f", output.shm.buf) == (42.0,) * _NUMEL
    finally:
        parent.close()
