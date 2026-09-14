# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real TMR ACLGraph numerics inside the C ABI test's isolated process.

The caller owns the prepared context, callable and tensor allocations. Exceptions
must terminate that process without unwinding graph-visible device resources.
"""

import ctypes
import struct

from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType

_CAPTURE_MODE_GLOBAL = 0
_COUNT = 128 * 128  # The compiled scalar-add child processes one 128x128 tile.
_SYNC_TIMEOUT_MS = 10000
_REPLAYS = 100


def _check(code, operation):
    assert code == 0, f"kernel_capture {operation}: rc={code}"


def run_capture_values(lib, ctx, caller, committed, allocations, scenario):
    signatures = {
        "aclmdlRICaptureBegin": [ctypes.c_void_p, ctypes.c_int],
        "aclmdlRICaptureEnd": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        "aclmdlRIExecuteAsync": [ctypes.c_void_p, ctypes.c_void_p],
        "aclmdlRIDestroy": [ctypes.c_void_p],
    }
    for symbol, argtypes in signatures.items():
        function = getattr(lib, symbol)
        function.argtypes = argtypes
        function.restype = ctypes.c_int

    host_array = ctypes.c_float * _COUNT
    tensor_bytes = ctypes.sizeof(host_array)
    poison = [-999.0] * _COUNT
    host_poison = host_array(*poison)
    scalars = (1.25, -3.5) if scenario == "capture_aba" else (1.25,)
    tensors = []
    for _ in scalars:
        pair = []
        for _ in range(2):
            address = ctypes.c_void_p()
            _check(lib.aclrtMalloc(ctypes.byref(address), tensor_bytes, 0), "allocate tensor")
            allocations.append(address)
            pair.append(address)
        tensors.append(pair)

    args = ChipStorageTaskArgs()
    last_outputs = [poison for _ in scalars]
    host_launches = 0

    def seed(index, iteration):
        values = [float(i % 127 + iteration * 257 + index * 4096) for i in range(_COUNT)]
        host_input = host_array(*values)
        source, destination = tensors[index]
        _check(lib.aclrtMemcpy(source, tensor_bytes, host_input, tensor_bytes, 1), "seed input")
        _check(lib.aclrtMemcpy(destination, tensor_bytes, host_poison, tensor_bytes, 1), "poison output")
        last_outputs[index] = poison
        return [value + scalars[index] for value in values]

    def verify(index, expected):
        host_output = host_array()
        _check(lib.aclrtMemcpy(host_output, tensor_bytes, tensors[index][1], tensor_bytes, 2), "read output")
        actual = list(host_output)
        if actual != expected:
            mismatch = next(i for i in range(_COUNT) if actual[i] != expected[i])
            raise AssertionError(
                f"kernel_capture graph={index} element={mismatch}: "
                f"actual={actual[mismatch]} expected={expected[mismatch]}"
            )

    def launch(index):
        nonlocal host_launches
        source, destination = tensors[index]
        args.clear()
        args.add_tensor(ChipTensor.make(source.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_tensor(ChipTensor.make(destination.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_scalar(int.from_bytes(struct.pack("<f", scalars[index]), "little"))
        _check(lib.simpler_kernel_mode_launch(ctx, 0, args.__ptr__(), caller), "public launch")
        host_launches += 1
        # A later capture reuses this metadata and the callable's encoder cache.
        args.clear()

    warmups = int(scenario == "capture_warm")
    if warmups:
        expected = seed(0, 0)
        launch(0)
        _check(lib.aclrtSynchronizeStreamWithTimeout(caller, _SYNC_TIMEOUT_MS), "warmup sync")
        verify(0, expected)

    graphs = []
    for index in range(len(scalars)):
        seed(index, 0)
        graph = ctypes.c_void_p()
        print(f"kernel_capture BEGIN scenario={scenario} graph={index} warmups={warmups}", flush=True)
        _check(lib.aclmdlRICaptureBegin(caller, _CAPTURE_MODE_GLOBAL), "capture begin")
        launch(index)
        _check(lib.aclmdlRICaptureEnd(caller, ctypes.byref(graph)), "capture end")
        assert graph.value, "capture returned a null graph"
        graphs.append(graph)
        assert lib.committed_device_memory_ctx(ctx) == committed

    for iteration in range(_REPLAYS):
        index = (0, 1, 0)[iteration % 3] if len(graphs) == 2 else 0
        expected = seed(index, iteration + 1)
        _check(lib.aclmdlRIExecuteAsync(graphs[index], caller), f"replay {iteration}")
        _check(lib.aclrtSynchronizeStreamWithTimeout(caller, _SYNC_TIMEOUT_MS), f"replay {iteration} sync")
        verify(index, expected)
        last_outputs[index] = expected
        for other in range(len(graphs)):
            if other != index:
                verify(other, last_outputs[other])
        assert lib.committed_device_memory_ctx(ctx) == committed

    assert host_launches == len(graphs) + warmups
    for graph in graphs:
        _check(lib.aclmdlRIDestroy(graph), "destroy graph before context/tensor release")
    print(
        f"kernel_capture PASS scenario={scenario} replays={_REPLAYS} graphs={len(graphs)} "
        f"warmups={warmups} host_launches={host_launches} committed={committed}",
        flush=True,
    )
