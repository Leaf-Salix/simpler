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

_CAPTURE_MODE_GLOBAL = 0
_COUNT = 128 * 128  # The compiled scalar-add child processes one 128x128 tile.
_SYNC_TIMEOUT_MS = 10000
_REPLAYS = 100
_POISON = -999.0


def batch_expected(sequence, initial=0.0):
    """Scalar oracle for two feedback graphs and their independent counters."""
    x, y = initial, _POISON
    counts = [0, 0]
    for index in sequence:
        if index == 0:
            y = x + 1.25
        else:
            assert index == 1
            x = y + 2.75
        counts[index] += 1
    return x, y, *counts


def _check(code, operation):
    assert code == 0, f"kernel_capture {operation}: rc={code}"


class _TensorIO:
    """Test tensor storage is owned and released by the subprocess caller."""

    def __init__(self, lib, allocations):
        self.lib = lib
        self.allocations = allocations
        self.array = ctypes.c_float * _COUNT
        self.nbytes = ctypes.sizeof(self.array)

    def allocate(self):
        address = ctypes.c_void_p()
        _check(self.lib.aclrtMalloc(ctypes.byref(address), self.nbytes, 0), "allocate tensor")
        self.allocations.append(address)
        return address

    def write(self, address, values):
        host_input = self.array(*values)
        _check(self.lib.aclrtMemcpy(address, self.nbytes, host_input, self.nbytes, 1), "seed tensor")

    def verify(self, address, expected):
        host_output = self.array()
        _check(self.lib.aclrtMemcpy(host_output, self.nbytes, address, self.nbytes, 2), "read output")
        actual = list(host_output)
        if actual != expected:
            mismatch = next(i for i in range(_COUNT) if actual[i] != expected[i])
            raise AssertionError(
                f"kernel_capture address={address.value:#x} element={mismatch}: "
                f"actual={actual[mismatch]} expected={expected[mismatch]}"
            )


def _bind_capture_functions(lib):
    observer = ctypes.CDLL(None)
    for symbol, return_type in {
        "capture_observer_begin": None,
        "capture_observer_status": ctypes.c_int,
        "capture_observer_check_resident": ctypes.c_int,
        "capture_observer_core_launches": ctypes.c_uint64,
        "capture_observer_cpu_launches": ctypes.c_uint64,
        "capture_observer_kernel_args": ctypes.c_uint64,
        "capture_observer_runtime_args": ctypes.c_uint64,
        "capture_observer_regs": ctypes.c_uint64,
    }.items():
        function = getattr(observer, symbol)
        function.argtypes = []
        function.restype = return_type
    observer.capture_observer_begin()
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
    return observer


def _check_resident(observer, host_launches):
    _check(observer.capture_observer_status(), "native launch observation")
    assert observer.capture_observer_core_launches() == host_launches
    assert observer.capture_observer_cpu_launches() == host_launches
    _check(observer.capture_observer_check_resident(), "resident address stability")


def run_capture_values(lib, ctx, caller, committed, allocations, scenario):
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    observer = _bind_capture_functions(lib)
    io = _TensorIO(lib, allocations)
    poison = [_POISON] * _COUNT
    args = ChipStorageTaskArgs()
    host_launches = 0
    syncs = 0

    def launch(source, destination, scalar):
        nonlocal host_launches
        args.clear()
        args.add_tensor(ChipTensor.make(source.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_tensor(ChipTensor.make(destination.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_scalar(int.from_bytes(struct.pack("<f", scalar), "little"))
        _check(lib.simpler_kernel_mode_launch(ctx, 0, args.__ptr__(), caller), "public launch")
        host_launches += 1
        # A later capture reuses this metadata and the callable's encoder cache.
        args.clear()

    def sync():
        nonlocal syncs
        _check(lib.aclrtSynchronizeStreamWithTimeout(caller, _SYNC_TIMEOUT_MS), "caller sync")
        syncs += 1

    graphs = []

    def record(nodes):
        graph = ctypes.c_void_p()
        print(f"kernel_capture BEGIN scenario={scenario} graph={len(graphs)} nodes={len(nodes)}", flush=True)
        _check(lib.aclmdlRICaptureBegin(caller, _CAPTURE_MODE_GLOBAL), "capture begin")
        for node in nodes:
            launch(*node)
        _check(lib.aclmdlRICaptureEnd(caller, ctypes.byref(graph)), "capture end")
        assert graph.value, "capture returned a null graph"
        graphs.append(graph)
        assert lib.committed_device_memory_ctx(ctx) == committed
        # Capture records tasks; prepare or the warmup sync establishes quiescence.
        _check_resident(observer, host_launches)

    def run_batch():
        x, y, count_a, count_b = [io.allocate() for _ in range(4)]
        initial = [float(i % 127) for i in range(_COUNT)]
        io.write(x, initial)
        io.write(y, poison)
        io.write(count_a, [0.0] * _COUNT)
        io.write(count_b, [0.0] * _COUNT)
        record([(x, y, 1.25), (count_a, count_a, 1.0)])
        record([(y, x, 2.75), (count_b, count_b, 1.0)])
        sequence = (0, 1) * (_REPLAYS // 2)
        expected_x, expected_y, expected_a, expected_b = batch_expected(sequence)
        # Only graph submission occurs in this loop: no copies, queries or waits.
        for index in sequence:
            _check(lib.aclmdlRIExecuteAsync(graphs[index], caller), "batch replay")
        sync()
        io.verify(x, [value + expected_x for value in initial])
        io.verify(y, [value + expected_y for value in initial])
        io.verify(count_a, [float(expected_a)] * _COUNT)
        io.verify(count_b, [float(expected_b)] * _COUNT)
        assert syncs == 1
        assert host_launches == 4

    warmups = int(scenario == "capture_warm")
    if scenario == "capture_batch":
        run_batch()
    elif scenario == "capture_chain":
        source, intermediate, destination = [io.allocate() for _ in range(3)]
        io.write(source, [0.0] * _COUNT)
        io.write(intermediate, poison)
        io.write(destination, poison)
        record([(source, intermediate, 1.25), (intermediate, destination, -3.5)])
        for iteration in range(_REPLAYS):
            values = [float(i % 127 + iteration * 257) for i in range(_COUNT)]
            io.write(source, values)
            io.write(intermediate, poison)
            io.write(destination, poison)
            _check(lib.aclmdlRIExecuteAsync(graphs[0], caller), f"chain replay {iteration}")
            sync()
            io.verify(intermediate, [value + 1.25 for value in values])
            io.verify(destination, [value - 2.25 for value in values])
            assert lib.committed_device_memory_ctx(ctx) == committed
        assert host_launches == 2
    else:
        scalars = (1.25, -3.5) if scenario == "capture_aba" else (1.25,)
        tensors = [(io.allocate(), io.allocate()) for _ in scalars]
        last_outputs = [poison for _ in scalars]

        def seed(index, iteration):
            values = [float(i % 127 + iteration * 257 + index * 4096) for i in range(_COUNT)]
            io.write(tensors[index][0], values)
            io.write(tensors[index][1], poison)
            last_outputs[index] = poison
            return [value + scalars[index] for value in values]

        if warmups:
            expected = seed(0, 0)
            launch(*tensors[0], scalars[0])
            sync()
            io.verify(tensors[0][1], expected)
        for index in range(len(scalars)):
            seed(index, 0)
            record([(*tensors[index], scalars[index])])
        for iteration in range(_REPLAYS):
            index = (0, 1, 0)[iteration % 3] if len(graphs) == 2 else 0
            expected = seed(index, iteration + 1)
            _check(lib.aclmdlRIExecuteAsync(graphs[index], caller), f"replay {iteration}")
            sync()
            io.verify(tensors[index][1], expected)
            last_outputs[index] = expected
            for other in range(len(graphs)):
                if other != index:
                    io.verify(tensors[other][1], last_outputs[other])
            assert lib.committed_device_memory_ctx(ctx) == committed
        assert host_launches == len(graphs) + warmups

    assert lib.committed_device_memory_ctx(ctx) == committed
    _check_resident(observer, host_launches)
    for graph in graphs:
        _check(lib.aclmdlRIDestroy(graph), "destroy graph before context/tensor release")
    print(
        f"kernel_capture PASS scenario={scenario} replays={_REPLAYS} graphs={len(graphs)} "
        f"warmups={warmups} host_launches={host_launches} syncs={syncs} committed={committed}",
        flush=True,
    )
    print(
        f"kernel_capture resident kernel_args={observer.capture_observer_kernel_args():#x} "
        f"runtime_args={observer.capture_observer_runtime_args():#x} "
        f"regs={observer.capture_observer_regs():#x} "
        f"native_core={observer.capture_observer_core_launches()} "
        f"native_cpu={observer.capture_observer_cpu_launches()}",
        flush=True,
    )
