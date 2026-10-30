# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the shared vector-add example without importing its Torch runner."""

from pathlib import Path

from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root


def build_chip_callable(platform):
    root = Path(__file__).resolve().parents[4] / "examples/workers/l3/multi_chip_dispatch/kernels"
    compiler = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    binary = compiler.compile_incore(
        source_path=str(root / "aiv/vector_add_kernel.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
    )
    if not platform.endswith("sim"):
        binary = extract_text_section(binary)
    signature = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]
    orchestration = compiler.compile_orchestration(
        runtime_name=runtime, source_path=str(root / "orchestration/vector_add_orch.cpp")
    )
    return ChipCallable.build(
        signature=signature,
        func_name="vector_add_orchestration",
        binary=orchestration,
        children=[(0, CoreCallable.build(signature=signature, binary=binary))],
    )
