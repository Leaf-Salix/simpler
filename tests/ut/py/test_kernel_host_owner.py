# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real binding lifetime checks with a non-device runtime fixture."""

import ctypes
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.kernel import KernelHostOwner, create_owner
from simpler.task_interface import CallConfig, ChipCallable, ChipStorageTaskArgs, ChipWorker


class Ticket(ctypes.Structure):
    _fields_ = [("opaque", ctypes.c_void_p), ("release", ctypes.c_void_p), ("invoke", ctypes.c_void_p)]


class Bridge(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("opaque", ctypes.c_void_p),
        ("accept", ctypes.c_void_p),
    ]


def _runtime_path():
    root = Path(__file__).resolve().parents[3]
    candidates = list((root / "tests/ut/cpp/build").glob("libkernel_host_test_runtime.*"))
    if not candidates:
        pytest.skip("build kernel_host_test_runtime first")
    return str(candidates[0])


@pytest.fixture
def kernel_runtime(tmp_path):
    path = _runtime_path()
    lib = ctypes.CDLL(path)
    for name in ("launches", "destroys", "finalizes", "creates", "prepares", "prepare_cid", "launch_cid"):
        function = getattr(lib, f"kernel_test_{name}")
        function.argtypes = []
        function.restype = ctypes.c_int
    for name in ("scalar", "generation"):
        function = getattr(lib, f"kernel_test_{name}")
        function.argtypes = []
        function.restype = ctypes.c_uint64
    for name in ("init_context", "prepare_context", "launch_context"):
        function = getattr(lib, f"kernel_test_{name}")
        function.argtypes = []
        function.restype = ctypes.c_size_t
    lib.kernel_test_errors.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.kernel_test_errors.restype = None
    lib.kernel_test_prepare_error.argtypes = [ctypes.c_int]
    lib.kernel_test_prepare_error.restype = None
    lib.kernel_test_reset.argtypes = []
    lib.kernel_test_reset.restype = None
    lib.kernel_test_reset()
    binary = tmp_path / "kernel.bin"
    binary.write_bytes(b"x")
    bins = SimpleNamespace(host_path=path, aicpu_path=binary, aicore_path=binary)
    yield lib, bins
    lib.kernel_test_errors(0, 0, 0)
    lib.kernel_test_prepare_error(0)


@pytest.fixture
def kernel_worker(kernel_runtime):
    lib, bins = kernel_runtime
    worker = ChipWorker()
    yield worker, lib, bins
    lib.kernel_test_errors(0, 0, 0)
    worker.finalize()


def _callable():
    return ChipCallable.build(signature=[], func_name="test", binary=b"x", children=[])


def _accept_ticket(owner, cid):
    capsule = owner._queue_handle()
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get.restype = ctypes.c_void_p
    bridge = Bridge.from_address(get(capsule, b"simpler.kernel_queue.v1"))
    accept = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(Ticket),
    )(bridge.accept)
    ticket = Ticket()
    assert accept(bridge.opaque, cid, None, 0, None, 0, ctypes.byref(ticket)) == 0
    return capsule, ticket


def _invoke_ticket(ticket):
    invoke = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ticket.invoke
    )
    return invoke(ticket.opaque, 1, None, None)


def _release_ticket(ticket):
    release = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(ticket.release)
    release(ticket.opaque)


def test_chip_worker_direct_and_queued_calls_share_context(kernel_worker):
    worker, lib, bins = kernel_worker
    worker.kernel_init(0, bins, CallConfig(), context_generation=123)
    assert worker.kernel_mode_supported
    assert lib.kernel_test_generation() == 123
    assert lib.kernel_test_creates() == 1
    cid = worker.kernel_prepare_callable(_callable())
    assert worker.committed_device_memory == 4096
    with pytest.raises(RuntimeError, match="program-owned"):
        worker.device_memory_info()
    assert lib.kernel_test_prepares() == 1
    assert lib.kernel_test_prepare_cid() == cid
    assert lib.kernel_test_prepare_context() == lib.kernel_test_init_context() != 0
    args = ChipStorageTaskArgs()
    args.add_scalar(41)
    worker.kernel_launch(cid, args, 1)
    assert lib.kernel_test_scalar() == 41
    assert lib.kernel_test_launch_cid() == cid
    assert lib.kernel_test_launch_context() == lib.kernel_test_init_context()
    capsule, ticket = _accept_ticket(worker, cid)
    try:
        assert _invoke_ticket(ticket) == 0
        assert lib.kernel_test_launches() == 2
        assert lib.kernel_test_launch_context() == lib.kernel_test_init_context()
        assert lib.kernel_test_creates() == 1
    finally:
        _release_ticket(ticket)
        del capsule
    worker.finalize()
    assert lib.kernel_test_destroys() == 1
    assert worker.committed_device_memory == 0


def test_chip_worker_pending_finalize_preserves_registration(kernel_worker):
    worker, lib, bins = kernel_worker
    worker.kernel_init(0, bins, CallConfig())
    cid = worker.kernel_prepare_callable(_callable())
    capsule, ticket = _accept_ticket(worker, cid)
    try:
        with pytest.raises(RuntimeError):
            worker.finalize()
        assert lib.kernel_test_finalizes() == 0
        assert lib.kernel_test_destroys() == 0
        assert cid in worker._callable_registry
        with pytest.raises(RuntimeError):
            worker.kernel_prepare_callable(_callable())
        assert lib.kernel_test_prepares() == 1
        assert len(worker._callable_registry) == 1
        assert _invoke_ticket(ticket) == 0
    finally:
        _release_ticket(ticket)
        del capsule
    worker.finalize()
    assert lib.kernel_test_finalizes() == 1
    assert lib.kernel_test_destroys() == 1
    assert not worker._callable_registry


@pytest.mark.parametrize("init_error", [0, -1001])
def test_chip_worker_finalize_failure_is_retryable(kernel_worker, init_error):
    worker, lib, bins = kernel_worker
    lib.kernel_test_errors(init_error, -1003, 0)
    if init_error:
        with pytest.raises(RuntimeError, match=str(init_error)):
            worker.kernel_init(0, bins, CallConfig())
        with pytest.raises(RuntimeError):
            worker.init(0, bins)
        with pytest.raises(RuntimeError):
            worker.kernel_init(0, bins, CallConfig())
        assert lib.kernel_test_creates() == 1
    else:
        worker.kernel_init(0, bins, CallConfig())
        worker.kernel_prepare_callable(_callable())
    with pytest.raises(RuntimeError, match="-1003"):
        worker.finalize()
    assert lib.kernel_test_destroys() == 0
    assert lib.kernel_test_finalizes() == 1
    assert len(worker._callable_registry) == (0 if init_error else 1)
    assert worker.committed_device_memory == 4096
    with pytest.raises(RuntimeError):
        worker.init(0, bins)
    with pytest.raises(RuntimeError):
        worker.kernel_init(0, bins, CallConfig())
    assert lib.kernel_test_creates() == 1
    lib.kernel_test_errors(0, 0, 0)
    worker.finalize()
    assert lib.kernel_test_destroys() == 1
    assert lib.kernel_test_finalizes() == 2
    assert not worker._callable_registry


def test_chip_worker_prepare_failure_reuses_unpublished_slot(kernel_worker):
    worker, lib, bins = kernel_worker
    worker.kernel_init(0, bins, CallConfig())
    lib.kernel_test_prepare_error(-1003)
    with pytest.raises(RuntimeError, match="-1003"):
        worker.kernel_prepare_callable(_callable())
    failed_cid = lib.kernel_test_prepare_cid()
    assert not worker._callable_registry
    lib.kernel_test_prepare_error(0)
    assert worker.kernel_prepare_callable(_callable()) == failed_cid
    assert lib.kernel_test_prepares() == 2


@pytest.mark.parametrize("repeat_mode", ["program", "kernel"])
def test_failed_kernel_init_keeps_original_close_thread(kernel_worker, repeat_mode):
    worker, lib, bins = kernel_worker
    lib.kernel_test_errors(-1000, 0, 0)
    with pytest.raises(RuntimeError, match="-1000"):
        worker.kernel_init(0, bins, CallConfig())

    def repeat_init():
        with pytest.raises(RuntimeError):
            if repeat_mode == "program":
                worker.init(0, bins)
            else:
                worker.kernel_init(0, bins, CallConfig())

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(repeat_init).result()
    assert lib.kernel_test_creates() == 1
    worker.finalize()
    assert lib.kernel_test_finalizes() == 1
    assert lib.kernel_test_destroys() == 1


def test_chip_worker_native_launch_error_is_not_swallowed(kernel_worker):
    worker, lib, bins = kernel_worker
    worker.kernel_init(0, bins, CallConfig())
    cid = worker.kernel_prepare_callable(_callable())
    lib.kernel_test_errors(0, 0, -1003)
    with pytest.raises(RuntimeError, match="-1003"):
        worker.kernel_launch(cid, ChipStorageTaskArgs(), 1)
    assert lib.kernel_test_launches() == 1
    worker.finalize()
    assert lib.kernel_test_destroys() == 1


def test_chip_worker_launch_requires_explicit_nonzero_stream(kernel_worker):
    worker, lib, bins = kernel_worker
    worker.kernel_init(0, bins, CallConfig())
    cid = worker.kernel_prepare_callable(_callable())
    with pytest.raises(TypeError):
        worker.kernel_launch(cid, ChipStorageTaskArgs())
    with pytest.raises(ValueError, match="caller_stream"):
        worker.kernel_launch(cid, ChipStorageTaskArgs(), 0)
    assert lib.kernel_test_launches() == 0


def test_chip_worker_automatic_generation_is_unique(kernel_runtime):
    lib, bins = kernel_runtime
    workers = [ChipWorker(), ChipWorker()]
    try:
        workers[0].kernel_init(0, bins, CallConfig())
        first = lib.kernel_test_generation()
        workers[1].kernel_init(0, bins, CallConfig())
        assert lib.kernel_test_generation() > first > 0
        assert lib.kernel_test_creates() == 2
    finally:
        for worker in workers:
            worker.finalize()


def test_native_ticket_release_never_needs_gil():
    path = _runtime_path()
    owner = create_owner(path)
    assert isinstance(owner, KernelHostOwner)
    assert owner.initialize(0, b"x", b"x", b"", CallConfig(), 1) == 0
    capsule = owner._queue_handle()
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get.restype = ctypes.c_void_p
    bridge = Bridge.from_address(get(capsule, b"simpler.kernel_queue.v1"))
    accept = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(Ticket),
    )(bridge.accept)
    ticket = Ticket()
    assert accept(bridge.opaque, 0, None, 0, None, 0, ctypes.byref(ticket)) == 0
    assert owner.pending == 1
    invoke = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ticket.invoke
    )
    assert invoke(ticket.opaque, 1, None, None) == 0
    assert owner.close() == 0
    del capsule, owner
    gc.collect()
    # PyDLL holds the GIL while the C++ helper joins the releasing thread.
    # A lease with a Python deleter would deadlock here.
    lib = ctypes.PyDLL(path)
    release = lib.kernel_test_release_on_thread
    release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    release(ticket.opaque, ticket.release)
