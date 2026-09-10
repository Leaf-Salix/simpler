# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit Host ownership and optional Torch-NPU launch for L2 kernel mode.

Close requires external device quiescence and destruction of every graph
referencing the context. In-flight Tensor storage/metadata must not be replaced.
The adapter records default caching-allocator storage, not arbitrary allocations.
"""

import importlib


def __getattr__(name: str):
    if name != "KernelHostOwner":
        raise AttributeError(name)
    # Use the existing source/binding revision check before exposing the type.
    importlib.import_module("simpler.task_interface")
    return importlib.import_module("_task_interface").KernelHostOwner


def create_owner(runtime_path: str):
    """Load a kernel runtime and create an uninitialized owner; no Torch required."""
    importlib.import_module("simpler.task_interface")
    return importlib.import_module("_task_interface").create_kernel_host_owner(runtime_path)


def enqueue(owner, callable_id: int, tensors, scalar_bits=(), *, op_name: str = "simpler_kernel") -> None:
    """Queue on a kernel-mode ChipWorker or a low-level KernelHostOwner.

    Scalars are explicit uint64 bit patterns. A callable id is valid only on
    the owner/worker where it was prepared; this call never creates a context.

    Input and output tensors must all be included in signature order. This call
    does not synchronize. Async native failures are reported by Torch-NPU.

    Native execution, including partial-enqueue failure cleanup, must join all
    internal use back to the caller stream. pending == 0 only means Host
    callbacks have returned, not Device completion or graph destruction.

    Recognizable external allocations are rejected. Extra logical ownership
    (for example a framework slot carved from ordinary Torch storage) cannot be
    inferred from a Tensor: the caller must retain that owner's lease and forbid
    slot reuse until all Device/graph uses finish. Tensor retention/recordStream
    alone do not protect such a slot from its owner's explicit reuse.
    """
    try:
        torch = importlib.import_module("torch")
        torch_npu = importlib.import_module("torch_npu")
        adapter = importlib.import_module("_torch_npu_adapter")
    except ImportError as exc:
        raise RuntimeError(
            "Simpler was built without a usable Torch-NPU adapter; rebuild with matching dependencies"
        ) from exc

    if adapter.BUILD_TORCH_VERSION != torch.__version__ or adapter.BUILD_TORCH_NPU_VERSION != torch_npu.__version__:
        raise RuntimeError("Simpler Torch-NPU adapter build/runtime versions do not match")
    adapter.enqueue(owner._queue_handle(), callable_id, tensors, scalar_bits, op_name)
