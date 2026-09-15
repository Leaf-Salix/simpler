# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cold capture of the public asynchronous TMR preparation/launch contract."""

import ctypes
import os
import platform
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.ut.py.test_kernel_mode_c_api import CallConfig, _binaries, _build_eager_callable, _load

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = "tensormap_and_ringbuffer"
SCENARIOS = (
    "cold_synced",
    "warm",
    "multi_callable",
    "cross_stream",
    "prepare_again",
    "query_error",
    "pending_same",
    "pending_cross",
    "replay_stream",
    "two_graphs",
    "prepare_fail_register",
    "prepare_fail_record",
    "prepare_fail_wait",
)


@pytest.fixture(scope="module")
def capture_observer(tmp_path_factory):
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    include_dirs = [
        sdk / "include",
        sdk / f"{platform.machine()}-linux/pkg_inc",
        sdk / f"{platform.machine()}-linux/pkg_inc/runtime",
        sdk / f"{platform.machine()}-linux/pkg_inc/runtime/runtime",
        sdk / f"{platform.machine()}-linux/pkg_inc/profiling",
        ROOT / "src/a2a3/platform/include",
        ROOT / "src/common/platform/include",
        ROOT / "src/common",
    ]
    output = tmp_path_factory.mktemp("cold-capture-observer") / "observer.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            *[f"-I{path}" for path in include_dirs],
            str(Path(__file__).with_name("kernel_capture_observer.cpp")),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_async_prepare_cold_capture(request, scenario, capture_observer):
    _binaries("a2a3", RUNTIME)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(capture_observer) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, "-m", "tests.ut.py.test_kernel_cold_capture", device, scenario],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if scenario.startswith("prepare_fail_"):
        assert f"PASS {scenario} rejected_after_failure=1 forbidden_sync=0" in result.stdout
    else:
        assert f"PASS {scenario} replays=100 forbidden_sync=0" in result.stdout


def _alternate_callable():
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler("a2a3")
    with tempfile.TemporaryDirectory(prefix="kernel-capture-") as build_dir:
        orchestration = compiler.compile_orchestration(
            RUNTIME, str(Path(__file__).with_name("kernel_capture_alternate.cpp")), build_dir=build_dir
        )
        incore = compiler.compile_incore(
            str(ROOT / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
            build_dir=build_dir,
        )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature, func_name="kernel_capture_alternate", binary=orchestration, children=[(0, child)]
    )


def _bind_acl(lib):
    signatures = {
        "aclInit": [ctypes.c_char_p],
        "aclFinalize": [],
        "aclrtSynchronizeDevice": [],
        "aclrtSetDevice": [ctypes.c_int],
        "aclrtResetDevice": [ctypes.c_int],
        "aclrtCreateStream": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtDestroyStream": [ctypes.c_void_p],
        "aclrtSynchronizeStreamWithTimeout": [ctypes.c_void_p, ctypes.c_int32],
        "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
        "aclrtFree": [ctypes.c_void_p],
        "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    }
    for name, arguments in signatures.items():
        getattr(lib, name).argtypes = arguments
        getattr(lib, name).restype = ctypes.c_int


def _config():
    config = CallConfig()
    for ring in range(4):
        config.runtime_env[ring] = 64
        config.runtime_env[ring + 4] = 1 << 20
        config.runtime_env[ring + 8] = 1024
    return config


def _bind_observer_guards(observer):
    observer.capture_observer_guard_sync.argtypes = [ctypes.c_int]
    observer.capture_observer_guard_sync.restype = None
    observer.capture_observer_invocation_scope.argtypes = [ctypes.c_int]
    observer.capture_observer_invocation_scope.restype = None
    observer.capture_observer_sync_calls.argtypes = []
    observer.capture_observer_sync_calls.restype = ctypes.c_uint64
    observer.capture_observer_override_query.argtypes = [ctypes.c_int]
    observer.capture_observer_override_query.restype = None
    observer.capture_observer_fail_prepare.argtypes = [ctypes.c_int]
    observer.capture_observer_fail_prepare.restype = None
    for name in ("query_calls", "total_queries", "waits", "records", "clears", "prepare_waits", "prepare_failures"):
        function = getattr(observer, "capture_observer_" + name)
        function.argtypes = []
        function.restype = ctypes.c_uint64


def _submission_counts(observer):
    return tuple(
        getattr(observer, "capture_observer_" + name)()
        for name in (
            "waits",
            "records",
            "clears",
            "core_launches",
            "cpu_launches",
        )
    )


def _check_steady_launch(observer, launch, *arguments, **keywords):
    before = observer.capture_observer_total_queries()
    launch(*arguments, **keywords)
    assert observer.capture_observer_total_queries() == before, "steady same-caller launch queried an old event"


def _check_query_branch(scenario, observer, launch):
    if scenario == "query_error":
        before = _submission_counts(observer)
        observer.capture_observer_override_query(1)
        launch(0, expected=-4332)
        assert observer.capture_observer_query_calls() == 1
        assert _submission_counts(observer) == before, "query error submitted device work"
        launch(0)
    elif scenario in ("pending_same", "pending_cross"):
        observer.capture_observer_override_query(2)
        launch(0)
        assert observer.capture_observer_query_calls() == 1
        assert observer.capture_observer_prepare_waits() == 1, "pending prepare lost its native event wait"


def _close(lib, ctx, allocations, streams, device):
    from tests.ut.py.kernel_capture_values import _check  # noqa: PLC0415

    _check(lib.finalize_device(ctx), "finalize context")
    assert lib.committed_device_memory_ctx(ctx) == 0
    lib.destroy_device_context(ctx)
    for address in reversed(allocations):
        _check(lib.aclrtFree(address), "free tensor")
    for stream in streams:
        _check(lib.aclrtDestroyStream(stream), "destroy stream")
    _check(lib.aclrtResetDevice(device), "reset")
    _check(lib.aclFinalize(), "finalize ACL")


def _fail(scenario, error):
    import traceback  # noqa: PLC0415

    traceback.print_exc()
    if scenario == "cold_unsynced":
        print(f"DIAGNOSTIC cold_unsynced failure={error}", flush=True)
    sys.stderr.flush()
    # Failed enqueue/capture does not establish graph-visible resource quiescence.
    os._exit(1)


def _check_prepare_failure(scenario, observer, lib, ctx, prepare, launch):
    from tests.ut.py.kernel_capture_values import _check  # noqa: PLC0415

    kind = {"prepare_fail_register": 1, "prepare_fail_record": 2, "prepare_fail_wait": 3}[scenario]
    before = lib.committed_device_memory_ctx(ctx)
    observer.capture_observer_fail_prepare(kind)
    prepare(0, expected=-4333)
    assert observer.capture_observer_prepare_failures() == 1
    retained = lib.committed_device_memory_ctx(ctx)
    assert retained > before
    prepare(0, expected=-1003)
    submission = _submission_counts(observer)
    launch(0, expected=-1003)
    assert _submission_counts(observer) == submission
    assert observer.capture_observer_prepare_failures() == 1
    assert lib.committed_device_memory_ctx(ctx) == retained
    # A failed caller wait cannot cover work already queued on the hidden stream.
    _check(lib.aclrtSynchronizeDevice(), "external drain before poisoned-context close")
    assert lib.committed_device_memory_ctx(ctx) == retained


def _seed_tensors(io, chips):
    from tests.ut.py.kernel_capture_values import _COUNT  # noqa: PLC0415

    pairs = [(io.allocate(), io.allocate()) for _ in chips]
    initial = [float(i % 127) for i in range(_COUNT)]
    for source, destination in pairs:
        io.write(source, initial)
        io.write(destination, [-999.0] * _COUNT)
    counter = io.allocate()
    io.write(counter, [0.0] * _COUNT)
    return pairs, initial, counter


def _run(device, scenario):
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    from tests.ut.py.kernel_capture_values import (  # noqa: PLC0415
        _COUNT,
        _bind_capture_functions,
        _check,
        _check_resident,
        _TensorIO,
    )

    chips = [_build_eager_callable("a2a3", RUNTIME)]
    if scenario in ("multi_callable", "prepare_again"):
        chips.append(_alternate_callable())
    lib = _load("a2a3", "onboard", RUNTIME)
    _bind_acl(lib)
    _check(lib.aclInit(None), "acl init")
    _check(lib.aclrtSetDevice(device), "device")
    streams = [ctypes.c_void_p(), ctypes.c_void_p()]
    for stream in streams:
        _check(lib.aclrtCreateStream(ctypes.byref(stream)), "create stream")
    caller = streams[int(scenario in ("cross_stream", "pending_cross"))]
    ctx = lib.create_device_context()
    assert ctx
    config = _config()
    aicpu, aicore, dispatcher = _binaries("a2a3", RUNTIME)
    _check(
        lib.simpler_kernel_mode_init(
            ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 71
        ),
        "kernel init",
    )
    observer = _bind_capture_functions(lib)
    _bind_observer_guards(observer)
    allocations = []
    io = _TensorIO(lib, allocations)
    pairs, initial, counter = _seed_tensors(io, chips)
    args = ChipStorageTaskArgs()
    host_launches = 0
    graphs = []

    def guarded(operation, *arguments, expected=0):
        observer.capture_observer_guard_sync(1)
        try:
            result = operation(*arguments)
        finally:
            observer.capture_observer_guard_sync(0)
        assert observer.capture_observer_sync_calls() == 0, "prepare/launch performed an internal sync"
        assert result == expected, f"guarded native operation rc={result}, expected={expected}"

    def prepare(cid, expected=0):
        chip = chips[cid]
        guarded(
            lib.simpler_kernel_mode_prepare_callable,
            ctx,
            cid,
            chip.buffer_ptr(),
            chip.buffer_size(),
            streams[0],
            expected=expected,
        )

    def sync(stream):
        _check(lib.aclrtSynchronizeStreamWithTimeout(stream, 10000), "external sync")

    def launch(cid, tensors=None, expected=0, scalar=1.25):
        nonlocal host_launches
        source, destination = pairs[cid] if tensors is None else tensors
        args.clear()
        args.add_tensor(ChipTensor.make(source.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_tensor(ChipTensor.make(destination.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_scalar(int.from_bytes(struct.pack("<f", scalar), "little"))
        observer.capture_observer_invocation_scope(1)
        try:
            guarded(lib.simpler_kernel_mode_launch, ctx, cid, args.__ptr__(), caller, expected=expected)
        finally:
            observer.capture_observer_invocation_scope(0)
        args.clear()
        host_launches += int(expected == 0)

    def record(cids, increment=1.25):
        graph = ctypes.c_void_p()
        _check(lib.aclmdlRICaptureBegin(caller, 0), "capture begin")
        launch(cids[0])
        for cid in cids[1:]:
            _check_steady_launch(observer, launch, cid)
        _check_steady_launch(observer, launch, 0, (counter, counter), scalar=increment)
        _check(lib.aclmdlRICaptureEnd(caller, ctypes.byref(graph)), "capture end")
        assert graph.value
        graphs.append(graph)

    try:
        if scenario.startswith("prepare_fail_"):
            _check_prepare_failure(scenario, observer, lib, ctx, prepare, launch)
            _close(lib, ctx, allocations, streams, device)
            print(f"PASS {scenario} rejected_after_failure=1 forbidden_sync=0", flush=True)
            return
        prepare(0)
        if scenario == "multi_callable":
            prepare(1)
        if scenario != "cold_unsynced":
            sync(streams[0])
        if scenario in ("query_error", "pending_same", "pending_cross"):
            _check_query_branch(scenario, observer, launch)
            sync(caller)
        if scenario in ("warm", "prepare_again"):
            launch(0)
            sync(caller)
        if scenario == "prepare_again":
            prepare(1)
            sync(streams[0])
        committed = lib.committed_device_memory_ctx(ctx)
        record([0, 1, 0] if len(chips) == 2 else [0])
        if scenario == "two_graphs":
            record([0], increment=2.5)
        if scenario != "cold_unsynced":
            _check_resident(observer, host_launches)
        replay_stream = streams[1] if scenario == "replay_stream" else caller
        for iteration in range(100):
            _check(lib.aclmdlRIExecuteAsync(graphs[iteration % len(graphs)], replay_stream), "replay")
        sync(replay_stream)
        for cid, (_, destination) in enumerate(pairs):
            io.verify(destination, [value + (1.25 if cid == 0 else -1.25) for value in initial])
        io.verify(counter, [187.5 if scenario == "two_graphs" else 125.0] * _COUNT)
        _check_resident(observer, host_launches)
        assert lib.committed_device_memory_ctx(ctx) == committed
        for graph in graphs:
            _check(lib.aclmdlRIDestroy(graph), "destroy graph")
        _close(lib, ctx, allocations, streams, device)
    except BaseException as error:
        _fail(scenario, error)
    print(f"PASS {scenario} replays=100 forbidden_sync=0 host_launches={host_launches}", flush=True)


if __name__ == "__main__":
    _run(int(sys.argv[1]), sys.argv[2])
