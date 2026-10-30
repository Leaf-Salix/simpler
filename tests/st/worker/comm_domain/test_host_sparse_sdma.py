# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-selected Host rows via SDMA, with an independent task and dependent consumer."""

import struct
from pathlib import Path

import pytest
from simpler.buffer import AccessMode
from simpler.task_interface import (
    ArgDirection as D,
)
from simpler.task_interface import (
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    HostWindowSpec,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
)
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

N = 128 * 128
SIZE = N * 4
RUNTIME = "tensormap_and_ringbuffer"
ROOT = Path(__file__).resolve().parents[4]


def build_sparse_sdma(platform):
    compiler = KernelCompiler(platform=platform)
    include = compiler.get_orchestration_include_dirs(RUNTIME)
    sources = [
        Path(__file__).parent / "kernels/host_sparse_sdma.cpp",
        ROOT / "examples/a2a3/tensormap_and_ringbuffer/sdma_async_completion_demo/kernels/aiv/kernel_consumer.cpp",
    ]
    children = []
    for index, source in enumerate(sources):
        binary = compiler.compile_incore(
            source_path=str(source), core_type="aiv", pto_isa_root=ensure_pto_isa_root(), extra_include_dirs=include
        )
        children.append(
            (
                index,
                CoreCallable.build(
                    signature=([D.IN, D.IN, D.OUT] if index == 0 else [D.IN, D.OUT]),
                    binary=extract_text_section(binary),
                ),
            )
        )
    orchestration = compiler.compile_orchestration(
        runtime_name=RUNTIME, source_path=str(Path(__file__).parent / "kernels/host_sparse_sdma_orch.cpp")
    )
    return ChipCallable.build(
        signature=[D.IN, D.IN, D.IN, D.OUT, D.OUT, D.OUT],
        func_name="host_sparse_sdma",
        binary=orchestration,
        children=children,
    )


@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.sdma
def test_host_sparse_sdma_consumer(st_platform, st_device_ids):
    # Trace this DAG to measure overlap; numerical success alone proves neither
    # overlap nor a speedup. Output allocation is one row, not the full table.
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=RUNTIME,
        device_ids=list(map(int, st_device_ids[:2])),
        num_sub_workers=0,
        enable_sdma=True,
    )
    chip = worker.register(build_sparse_sdma(st_platform))
    domains = []
    try:
        worker.init()
        ids = worker.create_buffer(4)
        independent = worker.create_buffer(SIZE)
        consumed = worker.create_buffer(SIZE)
        independent_result = worker.create_buffer(SIZE)
        independent.shm.buf[:SIZE] = struct.pack(f"<{N}f", *([11.0] * N))
        for selected in (0, 127, 7, 7):
            ids.shm.buf[:4] = struct.pack("<I", selected)

            def dispatch(orch, _args, cfg):
                domain = orch.get_or_allocate_domain(
                    name="sparse",
                    members=[0, 1],
                    window_size=SIZE,
                    buffers=[CommBufferSpec("payload", "float32", N, SIZE)],
                    host_window=HostWindowSpec(128 * SIZE),
                    retain_after_run=True,
                )
                if not domains:
                    domains.append(domain)
                    for row in range(128):
                        orch.copy_to_host_domain(domain, struct.pack(f"<{N}f", *([float(row)] * N)), offset=row * SIZE)
                else:
                    assert domain is domains[0]
                table = domain.map_view(
                    domain.host().tensor((128 * N,), DataType.FLOAT32), domain_rank=1, access=AccessMode.READ
                )
                args = TaskArgs()
                args.add_tensor(table, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
                args.add_tensor(ids.tensor((1,), DataType.UINT32), TensorArgType.INPUT)
                args.add_tensor(independent.tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
                args.add_tensor(
                    domain.member(1).buffers["payload"].tensor((N,), DataType.FLOAT32),
                    TensorArgType.OUTPUT_EXISTING,
                    transfer=TensorTransfer.NONE,
                )
                args.add_tensor(consumed.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(independent_result.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                orch.submit_next_level(chip, args, cfg, worker=1)

            worker.run(dispatch)
            assert struct.unpack_from(f"<{N}f", consumed.shm.buf) == (float(selected + 1),) * N
            assert struct.unpack_from(f"<{N}f", independent_result.shm.buf) == (12.0,) * N
        domains[0].release()
        assert domains[0].freed
    finally:
        worker.close()
