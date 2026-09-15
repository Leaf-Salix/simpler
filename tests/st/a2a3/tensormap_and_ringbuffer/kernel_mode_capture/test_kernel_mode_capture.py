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
from pathlib import Path

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.graph_cases import run_graph_case
from tests.ut.py.test_kernel_mode_c_api import CallConfig, _binaries, _load

ROOT = Path(__file__).resolve().parents[5]
MODULE = "tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture"
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
    "blocked_same",
    "blocked_cross",
    "replay_stream",
    "two_graphs",
    "fresh_inputs",
    "chain",
    "feedback_batch",
    "eager_replay",
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
            str(Path(__file__).with_name("prepare_gate.cpp")),
            "-pthread",
            f"-L{sdk / 'lib64'}",
            f"-Wl,-rpath,{sdk / 'lib64'}",
            "-lascendcl",
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
def test_async_prepare_cold_capture(st_platform, st_device_ids, scenario, capture_observer, tmp_path):
    _binaries(st_platform, RUNTIME)
    device = str(st_device_ids[0])
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(capture_observer) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    logs = tmp_path / "ascend"
    logs.mkdir()
    env["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    with (tmp_path / "run.log").open("w") as log:
        try:
            result = subprocess.run(
                [sys.executable, "-m", MODULE + ".test_kernel_mode_capture", device, scenario, str(tmp_path)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"capture scenario timed out; artifacts: {tmp_path}")
    output = (tmp_path / "run.log").read_text()
    assert result.returncode == 0, f"{output}\nArtifacts: {tmp_path}"
    if scenario.startswith("prepare_fail_"):
        assert f"PASS {scenario} rejected_after_failure=1 forbidden_sync=0" in output
    else:
        assert f"PASS {scenario} replays=100 forbidden_sync=0" in output


def _build_callable(build_dir, alternate=False):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler("a2a3")
    build_dir.mkdir()
    source = (
        Path(__file__).with_name("kernel_capture_alternate.cpp")
        if alternate
        else ROOT / "tests/ut/py/kernel_eager_orchestration.cpp"
    )
    orchestration = compiler.compile_orchestration(RUNTIME, str(source), build_dir=str(build_dir))
    incore = compiler.compile_incore(
        str(ROOT / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
        build_dir=str(build_dir),
    )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature,
        func_name="kernel_capture_alternate" if alternate else "kernel_eager_orchestration",
        binary=orchestration,
        children=[(0, child)],
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
    for name in ("arm", "release", "pending", "blocked", "finish"):
        function = getattr(observer, "capture_gate_" + name)
        function.argtypes = []
        function.restype = None if name in ("arm", "release") else ctypes.c_int
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


def _close(lib, ctx, allocations, streams, device, graphs=()):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    for graph in graphs:
        _check(lib.aclmdlRIDestroy(graph), "destroy graph")
    _check(lib.finalize_device(ctx), "finalize context")
    assert lib.committed_device_memory_ctx(ctx) == 0
    lib.destroy_device_context(ctx)
    for address in reversed(allocations):
        _check(lib.aclrtFree(address), "free tensor")
    for stream in streams:
        _check(lib.aclrtDestroyStream(stream), "destroy stream")
    _check(lib.aclrtResetDevice(device), "reset")
    _check(lib.aclFinalize(), "finalize ACL")


def _fail():
    import traceback  # noqa: PLC0415

    traceback.print_exc()
    sys.stderr.flush()
    # Failed enqueue/capture does not establish graph-visible resource quiescence.
    os._exit(1)


def _check_prepare_failure(scenario, observer, lib, ctx, prepare, launch):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

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
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _COUNT  # noqa: PLC0415

    pairs = [(io.allocate(), io.allocate()) for _ in chips]
    initial = [float(i % 127) for i in range(_COUNT)]
    for source, destination in pairs:
        io.write(source, initial)
        io.write(destination, [-999.0] * _COUNT)
    counter = io.allocate()
    io.write(counter, [0.0] * _COUNT)
    return pairs, initial, counter


def _initialize(device, scenario, build_dir):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (  # noqa: PLC0415
        _bind_capture_functions,
        _check,
    )

    chips = [_build_callable(build_dir / "callable-a")]
    if scenario in ("multi_callable", "prepare_again"):
        chips.append(_build_callable(build_dir / "callable-b", alternate=True))
    lib = _load("a2a3", "onboard", RUNTIME)
    _bind_acl(lib)
    _check(lib.aclInit(None), "acl init")
    _check(lib.aclrtSetDevice(device), "device")
    streams = [ctypes.c_void_p(), ctypes.c_void_p()]
    for stream in streams:
        _check(lib.aclrtCreateStream(ctypes.byref(stream)), "create stream")
    caller = streams[int(scenario in ("cross_stream", "pending_cross", "blocked_cross"))]
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
    return chips, lib, streams, caller, ctx, observer


def _prepare_initial(scenario, observer, prepare, launch):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    blocked = scenario in ("blocked_same", "blocked_cross")
    if blocked:
        observer.capture_gate_arm()
    prepare(0)
    if blocked:
        assert observer.capture_gate_blocked() == 1
        assert observer.capture_gate_pending() == 1
        before = observer.capture_observer_total_queries()
        launch(0)
        assert observer.capture_observer_total_queries() == before + 1
        assert observer.capture_gate_blocked() == 1, "launch waited for blocked prepare"
        assert observer.capture_gate_pending() == 1
        observer.capture_gate_release()
        _check(observer.capture_gate_finish(), "finish prepare gate")
    return blocked


def _verify_values(io, pairs, initial, counter, scenario):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _COUNT  # noqa: PLC0415

    for cid, (_, destination) in enumerate(pairs):
        io.verify(destination, [value + (1.25 if cid == 0 else -1.25) for value in initial])
    io.verify(counter, [187.5 if scenario == "two_graphs" else 125.0] * _COUNT)


def _run(device, scenario, build_dir):
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (  # noqa: PLC0415
        _COUNT,
        _check,
        _check_resident,
        _TensorIO,
    )

    chips, lib, streams, caller, ctx, observer = _initialize(device, scenario, build_dir)
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

    def record_nodes(nodes):
        graph = ctypes.c_void_p()
        _check(lib.aclmdlRICaptureBegin(caller, 0), "capture begin")
        for index, (cid, tensors, scalar) in enumerate(nodes):
            if index == 0:
                launch(cid, tensors, scalar=scalar)
            else:
                _check_steady_launch(observer, launch, cid, tensors, scalar=scalar)
        _check(lib.aclmdlRICaptureEnd(caller, ctypes.byref(graph)), "capture end")
        assert graph.value
        graphs.append(graph)
        return graph

    def record(cids, increment=1.25):
        return record_nodes([(cid, None, 1.25) for cid in cids] + [(0, (counter, counter), increment)])

    def replay(graph, stream=None):
        before = observer.capture_observer_total_queries()
        observer.capture_observer_invocation_scope(1)
        try:
            _check(lib.aclmdlRIExecuteAsync(graph, caller if stream is None else stream), "replay")
        finally:
            observer.capture_observer_invocation_scope(0)
        assert observer.capture_observer_total_queries() == before

    try:
        if scenario.startswith("prepare_fail_"):
            _check_prepare_failure(scenario, observer, lib, ctx, prepare, launch)
            _close(lib, ctx, allocations, streams, device)
            print(f"PASS {scenario} rejected_after_failure=1 forbidden_sync=0", flush=True)
            return
        if _prepare_initial(scenario, observer, prepare, launch):
            sync(caller)
            io.verify(pairs[0][1], [value + 1.25 for value in initial])
        if scenario == "multi_callable":
            prepare(1)
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
        if scenario in ("fresh_inputs", "chain", "feedback_batch", "eager_replay"):
            run_graph_case(scenario, io, record_nodes, replay, launch, lambda: sync(caller))
            _check_resident(observer, host_launches)
            assert lib.committed_device_memory_ctx(ctx) == committed
            _close(lib, ctx, allocations, streams, device, graphs)
            print(f"PASS {scenario} replays=100 forbidden_sync=0 host_launches={host_launches}", flush=True)
            return
        record([0, 1, 0] if len(chips) == 2 else [0])
        if scenario == "two_graphs":
            record([0], increment=2.5)
        _check_resident(observer, host_launches)
        replay_stream = streams[1] if scenario == "replay_stream" else caller
        for iteration in range(100):
            replay(graphs[iteration % len(graphs)], replay_stream)
        sync(replay_stream)
        _verify_values(io, pairs, initial, counter, scenario)
        _check_resident(observer, host_launches)
        assert lib.committed_device_memory_ctx(ctx) == committed
        _close(lib, ctx, allocations, streams, device, graphs)
    except BaseException:
        _fail()
    print(f"PASS {scenario} replays=100 forbidden_sync=0 host_launches={host_launches}", flush=True)


if __name__ == "__main__":
    _run(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
