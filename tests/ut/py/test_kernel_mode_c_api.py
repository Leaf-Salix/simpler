# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-mode C ABI driven through a real loaded host runtime.

The class-level tests cover the state machines in isolation; these cover the
glue no other test reaches — argument validation as the loaded component
actually performs it, and, on hardware, the bring-up and teardown of a real
kernel context.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PTO_RUNTIME_ERR_INTERNAL = -1000
PTO_RUNTIME_ERR_UNSUPPORTED = -1001
PTO_RUNTIME_ERR_INVALID_STATE = -1003

_ARCHES = ("a2a3", "a5")
_RUNTIMES = ("host_build_graph", "tensormap_and_ringbuffer")
_SIM_CASES = [pytest.param(arch, runtime, id=f"{arch}-sim-{runtime}") for arch in _ARCHES for runtime in _RUNTIMES]
_ONBOARD_CASES = [
    pytest.param(
        arch,
        runtime,
        id=f"{arch}-onboard-{runtime}",
        marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])],
    )
    for arch in _ARCHES
    for runtime in _RUNTIMES
]


@pytest.fixture(scope="module")
def kernel_close_faults(tmp_path_factory):
    output = tmp_path_factory.mktemp("kernel-close-faults") / "faults.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-I" + str(_PROJECT_ROOT / "src/common/log/include"),
            str(Path(__file__).with_name("kernel_close_faults.cpp")),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
@pytest.mark.parametrize(
    "scenario",
    ["repeat_init", "init_failure", "stream_close", "event_close", "destroy_unclosed", "prepare", "fatal_device"],
)
def test_kernel_lifecycle_retry(arch, runtime, scenario, kernel_close_faults, request):
    _binaries(arch, runtime)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, scenario],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if scenario == "destroy_unclosed":
        assert "refusing to destroy an unclosed kernel context" in result.stdout + result.stderr


# RUNTIME_ENV_FIELD_GROUPS(3) * RUNTIME_ENV_RING_COUNT(4); the size assertion
# below is what catches a layout change rather than this constant.
_RUNTIME_ENV_UINT64_FIELDS = 12
_OUTPUT_PREFIX_BYTES = 1024


class CallConfig(ctypes.Structure):
    """Mirror of the packed CallConfig the C ABI takes by pointer."""

    _pack_ = 1
    _fields_ = [
        ("aicpu_thread_num", ctypes.c_int32),
        ("enable_chip_swimlane", ctypes.c_int32),
        ("enable_dump_args", ctypes.c_int32),
        ("enable_pmu", ctypes.c_int32),
        ("enable_dep_gen", ctypes.c_int32),
        ("enable_scope_stats", ctypes.c_int32),
        ("capture_clock_anchors", ctypes.c_int32),
        ("runtime_env", ctypes.c_uint64 * _RUNTIME_ENV_UINT64_FIELDS),
        ("output_prefix", ctypes.c_char * _OUTPUT_PREFIX_BYTES),
    ]


def _load(arch: str, variant: str, runtime: str) -> ctypes.CDLL:
    path = _PROJECT_ROOT / "build" / "lib" / arch / variant / runtime / "libhost_runtime.so"
    if not path.exists():
        pytest.skip(f"{path} not built")
    if variant == "sim":
        # A simulated host runtime resolves its device entries against the
        # simulator context, which the worker normally loads for it.
        sim_context = _PROJECT_ROOT / "build" / "lib" / "libcpu_sim_context.so"
        if not sim_context.exists():
            pytest.skip(f"{sim_context} not built")
        ctypes.CDLL(str(sim_context), mode=ctypes.RTLD_GLOBAL)  # its hooks resolve by dlsym(RTLD_DEFAULT)
    # RTLD_LOCAL, as the worker loads it: two runtimes export the same entry
    # names, so a globally-scoped load makes the second one's calls land in
    # the first.
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.create_device_context.restype = ctypes.c_void_p
    lib.create_device_context.argtypes = []
    lib.destroy_device_context.argtypes = [ctypes.c_void_p]
    lib.finalize_device.argtypes = [ctypes.c_void_p]
    lib.finalize_device.restype = ctypes.c_int
    lib.committed_device_memory_ctx.argtypes = [ctypes.c_void_p]
    lib.committed_device_memory_ctx.restype = ctypes.c_size_t
    lib.simpler_kernel_mode_supported.argtypes = [ctypes.c_void_p]
    lib.simpler_kernel_mode_supported.restype = ctypes.c_int
    lib.simpler_kernel_mode_init.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(CallConfig), ctypes.c_uint64,
    ]  # fmt: skip
    lib.simpler_kernel_mode_init.restype = ctypes.c_int
    lib.simpler_kernel_mode_prepare_callable.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.simpler_kernel_mode_prepare_callable.restype = ctypes.c_int
    lib.simpler_kernel_mode_launch.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
    lib.simpler_kernel_mode_launch.restype = ctypes.c_int
    lib.simpler_init.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(CallConfig), ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64,
    ]  # fmt: skip
    lib.simpler_init.restype = ctypes.c_int
    return lib


def _minimal_callable_image() -> bytes:
    """A structurally valid ChipCallable image — the smallest input that gets
    past argument validation and reaches the ordering verdict."""
    import ctypes as _ctypes  # noqa: PLC0415

    from simpler.task_interface import ArgDirection, ChipCallable  # noqa: PLC0415

    chip = ChipCallable.build(signature=[ArgDirection.IN], func_name="f", binary=b"\x00", children=[])
    return _ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))


def _binaries(arch: str, runtime: str) -> tuple[bytes, bytes, bytes]:
    base = _PROJECT_ROOT / "build" / "lib" / arch / "onboard" / runtime
    dispatcher = _PROJECT_ROOT / "build" / "lib" / arch / "dispatcher" / "libsimpler_aicpu_dispatcher.so"
    files = (base / "libaicpu_kernel.so", base / "aicore_kernel.o", dispatcher)
    for path in files:
        if not path.exists():
            pytest.skip(f"{path} not built")
    return tuple(path.read_bytes() for path in files)  # type: ignore[return-value]


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_kernel_init_rejects_malformed_arguments(arch: str, runtime: str):
    """Every component runs the same structural validation before its verdict.

    The checks live in one shared header precisely so a stub and a real
    implementation accept and reject the same arguments; that parity is only
    provable against the loaded component.
    """
    lib = _load(arch, "sim", runtime)
    config = CallConfig()
    payload = b"\x00\x01\x02\x03"
    ctx = lib.create_device_context()
    assert ctx
    try:
        # A binary and its size describe one object, so a pointer without a
        # size — or a size without a pointer — is structurally invalid.
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, None, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_INTERNAL
        )
        # A negative device and a zero generation are both out of contract.
        for device_id, generation in ((-1, 1), (0, 0)):
            assert (
                lib.simpler_kernel_mode_init(
                    ctx,
                    device_id,
                    payload,
                    len(payload),
                    payload,
                    len(payload),
                    payload,
                    len(payload),
                    ctypes.byref(config),
                    generation,
                )  # fmt: skip
                == PTO_RUNTIME_ERR_INTERNAL
            )
        # A null config is rejected before anything else is read.
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), None, 1
            )
            == PTO_RUNTIME_ERR_INTERNAL
        )
    finally:
        lib.destroy_device_context(ctx)


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_kernel_init_rejects_invalid_static_config_without_claim(arch: str, runtime: str):
    lib = _load(arch, "sim", runtime)
    config = CallConfig()
    payload = b"\x00\x01\x02\x03"
    ctx = lib.create_device_context()
    assert ctx
    try:
        for threads in (-1, 1, 6):
            config.aicpu_thread_num = threads
            assert (
                lib.simpler_kernel_mode_init(
                    ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
                )
                == PTO_RUNTIME_ERR_INTERNAL
            )
        config.aicpu_thread_num = 0
        config.enable_dump_args = 1
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_UNSUPPORTED
        )
        config.enable_dump_args = 0
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_UNSUPPORTED
        )
        assert lib.committed_device_memory_ctx(ctx) == 0
        assert lib.simpler_kernel_mode_supported(ctx) == 0
    finally:
        lib.destroy_device_context(ctx)


def _run_lifecycle_retry(arch, runtime, device, scenario):
    lib = _load(arch, "onboard", runtime)
    # Caller-owned binding precedes kernel init and the forbidden-call window.
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.rtSetDevice.restype = ctypes.c_int
    assert lib.rtSetDevice(device) == 0
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    # Use an explicit, small-but-real TMR arena request for the onboard
    # lifecycle/prepare path.  Zeroed packed ring fields exercise the default
    # resolver, but do not provide a useful device descriptor for this
    # hardware smoke.
    for ring in range(4):
        config.runtime_env[ring] = 64
        config.runtime_env[4 + ring] = 1 << 20
        config.runtime_env[8 + ring] = 1024
    ctx = lib.create_device_context()
    init_args = (
        ctx,
        device,
        aicpu,
        len(aicpu),
        aicore,
        len(aicore),
        dispatcher,
        len(dispatcher),
        ctypes.byref(config),
        1,
    )
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    faults.bind_test_log.restype = ctypes.c_int
    assert faults.bind_test_log(lib._handle) == 0
    faults.arm_destroy_failure.argtypes = [ctypes.c_int]
    faults.destroy_attempts.restype = ctypes.c_int
    lib.ensure_acl_ready_ctx.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ensure_acl_ready_ctx.restype = ctypes.c_int
    faults.acl_call_count.argtypes = [ctypes.c_int]
    faults.acl_call_count.restype = ctypes.c_int
    faults.arm_acl_guard()
    # Prove each interceptor counts and refuses calls before trusting zeros.
    forbidden_args = (
        ("aclInit", ctypes.c_char_p, None),
        ("aclrtSetDevice", ctypes.c_int, device),
        ("aclrtResetDevice", ctypes.c_int, device),
        ("aclrtResetDeviceForce", ctypes.c_int, device),
        ("aclFinalize", None, None),
        ("rtDeviceReset", ctypes.c_int, device),
    )
    for i, (symbol, argtype, value) in enumerate(forbidden_args):
        fn = getattr(faults, symbol)
        fn.argtypes = [] if argtype is None else [argtype]
        fn.restype = ctypes.c_int
        assert (fn() if argtype is None else fn(value)) == -4322
        assert faults.acl_call_count(i) == 1
    faults.arm_acl_guard()
    if scenario == "init_failure":
        faults.arm_destroy_failure(3)
    assert lib.simpler_kernel_mode_init(*init_args) == (-4321 if scenario == "init_failure" else 0)
    try:
        # These exported C++ methods use the Linux Itanium ABI. Resolve the
        # actual runtime's methods rather than adding production test hooks.
        force_reset = getattr(lib, "_ZN12DeviceRunner18force_reset_deviceEv")
        force_reset.argtypes = [ctypes.c_void_p]
        force_reset.restype = ctypes.c_int
        assert force_reset(ctx) == PTO_RUNTIME_ERR_UNSUPPORTED
        assert lib.ensure_acl_ready_ctx(ctx, device) == PTO_RUNTIME_ERR_UNSUPPORTED
        if scenario == "fatal_device":
            recover = getattr(lib, "_ZN12DeviceRunner31recover_device_or_mark_unusableEi")
            recover.argtypes = [ctypes.c_void_p, ctypes.c_int]
            recover.restype = None
            accepts = getattr(lib, "_ZNK12DeviceRunner14can_accept_runEv")
            accepts.argtypes = [ctypes.c_void_p]
            accepts.restype = ctypes.c_bool
            assert accepts(ctx)
            # Inject the drain result, not a real device fault or device reset.
            # The real recovery method sets device_unusable_ and real finalize
            # must then take the fatal-device branch.
            faults.arm_fatal_drain()
            recover(ctx, 507018)
            assert faults.fatal_drain_count() == 1
            assert not accepts(ctx)
            assert force_reset(ctx) == PTO_RUNTIME_ERR_UNSUPPORTED
            assert lib.finalize_device(ctx) == 0
        elif scenario == "destroy_unclosed":
            lib.destroy_device_context(ctx)
            assert lib.ensure_acl_ready_ctx(ctx, device) == PTO_RUNTIME_ERR_UNSUPPORTED
            assert lib.finalize_device(ctx) == 0
        elif scenario == "prepare":
            # The caller's packed buffer stops being configuration authority
            # when init returns; prepare uses its owned, validated snapshot.
            config.aicpu_thread_num = -1
            config.enable_dump_args = 1
            _check_prepare_reuse(lib, ctx, arch, runtime)
        elif scenario in ("repeat_init", "init_failure"):
            assert lib.simpler_kernel_mode_init(*init_args) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.ensure_acl_ready_ctx(ctx, device) == PTO_RUNTIME_ERR_UNSUPPORTED
        else:
            faults.arm_destroy_failure(1 if scenario == "stream_close" else 2)
            assert lib.finalize_device(ctx) == -4321
            first_attempts = faults.destroy_attempts()
            assert first_attempts > 0
            assert lib.finalize_device(ctx) == 0
            assert faults.destroy_attempts() == first_attempts + 1
    finally:
        lib.finalize_device(ctx)
        lib.destroy_device_context(ctx)
        names = (
            "aclInit",
            "aclrtSetDevice",
            "aclrtResetDevice",
            "aclrtResetDeviceForce",
            "aclFinalize",
            "rtDeviceReset",
        )
        assert {name: faults.acl_call_count(i) for i, name in enumerate(names)} == dict.fromkeys(names, 0)


def _check_prepare_reuse(lib, ctx, arch, runtime):
    import tempfile  # noqa: PLC0415

    from simpler.task_interface import ChipCallable  # noqa: PLC0415

    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="kernel-prepare-") as build_dir:
        binary = KernelCompiler(arch).compile_orchestration(
            runtime, str(Path(__file__).with_name("kernel_prepare_orchestration.cpp")), build_dir=build_dir
        )
    chip = ChipCallable.build(signature=[], func_name="kernel_prepare_orchestration", binary=binary, children=[])
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    before = lib.committed_device_memory_ctx(ctx)
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
    prepared = lib.committed_device_memory_ctx(ctx)
    assert prepared > before
    # Duplicate registration is rejected; it must not disturb the first ID.
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) != 0
    assert lib.committed_device_memory_ctx(ctx) == prepared
    # Identical bytes deduplicate the callable upload. The second ID also
    # reuses the context's persistent argument blocks, so neither adds GM.
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 1, image, len(image)) == 0
    assert lib.committed_device_memory_ctx(ctx) == prepared
    assert lib.finalize_device(ctx) == 0
    assert lib.committed_device_memory_ctx(ctx) == 0
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 2, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_kernel_entries_reject_a_context_with_no_kernel_claim(arch: str, runtime: str):
    """Structurally valid calls on a context that never claimed kernel mode
    are an ordering error, not an argument error."""
    lib = _load(arch, "sim", runtime)
    image = _minimal_callable_image()
    stream = ctypes.byref((ctypes.c_uint8 * 8)())
    ctx = lib.create_device_context()
    assert ctx
    try:
        assert lib.simpler_kernel_mode_supported(ctx) == 0
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE
        assert lib.simpler_kernel_mode_launch(ctx, 0, image, stream) == PTO_RUNTIME_ERR_INVALID_STATE
        # An out-of-range callable id and a truncated image are argument
        # errors, so the structural checks run before the ordering one.
        assert lib.simpler_kernel_mode_prepare_callable(ctx, -1, image, len(image)) == PTO_RUNTIME_ERR_INTERNAL
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, 1) == PTO_RUNTIME_ERR_INTERNAL
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image) - 1) == PTO_RUNTIME_ERR_INTERNAL
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE
        assert lib.simpler_kernel_mode_launch(ctx, 0, image, None) == PTO_RUNTIME_ERR_INTERNAL
    finally:
        lib.destroy_device_context(ctx)


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_simulated_components_report_kernel_mode_unsupported(arch: str, runtime: str):
    """A component without kernel-mode execution still validates first, then
    reports unsupported — it never claims a mode it cannot honor."""
    lib = _load(arch, "sim", runtime)
    config = CallConfig()
    payload = b"\x00"
    ctx = lib.create_device_context()
    assert ctx
    try:
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_UNSUPPORTED
        )
        # The refused init took no claim, so the context is still free.
        image = _minimal_callable_image()
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE
    finally:
        lib.destroy_device_context(ctx)


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
def test_kernel_context_brings_up_and_closes_on_a_borrowed_device(arch: str, runtime: str, request):
    """Exercise the onboard kernel entry according to the runtime boundary.

    TMR owns the kernel contract in this stage and must bring up/close a
    borrowed context.  HBG has a link-complete kernel stub and must reject the
    same request without claiming the context.
    """
    lib = _load(arch, "onboard", runtime)
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    device_id = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.rtSetDevice.restype = ctypes.c_int
    assert lib.rtSetDevice(device_id) == 0

    ctx = lib.create_device_context()
    assert ctx
    try:
        rc = (
            lib.simpler_kernel_mode_init(
                ctx,
                device_id,
                aicpu,
                len(aicpu),
                aicore,
                len(aicore),
                dispatcher,
                len(dispatcher),
                ctypes.byref(config),
                1,
            )  # fmt: skip
        )
        if runtime == "host_build_graph":
            # HBG's kernel builder is intentionally a link-complete stub in
            # this stage; only the TMR runtime owns the kernel contract.
            assert rc == PTO_RUNTIME_ERR_UNSUPPORTED
            return
        assert rc == 0
        # The claim is exclusive for the context's whole life.
        assert (
            lib.simpler_init(
                ctx,
                device_id,
                aicpu,
                len(aicpu),
                aicore,
                len(aicore),
                dispatcher,
                len(dispatcher),
                ctypes.byref(config),
                0,
                None,
                0,
            )  # fmt: skip
            == PTO_RUNTIME_ERR_INVALID_STATE
        )
        assert lib.finalize_device(ctx) == 0
    finally:
        # Destroying after a clean close is allowed; an unclosed kernel
        # context would be refused and leaked instead.
        lib.destroy_device_context(ctx)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
def test_kernel_eager_launch_executes_fresh_tensor_and_scalar_snapshots(request):
    """Run one real TMR AIV callable through the public kernel C ABI."""
    import struct
    import tempfile

    from simpler.task_interface import ArgDirection, ChipStorageTaskArgs, ChipTensor, CoreCallable, ChipCallable, DataType
    from simpler_setup.elf_parser import extract_text_section
    from simpler_setup.kernel_compiler import KernelCompiler
    from simpler_setup.pto_isa import ensure_pto_isa_root

    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    root = _PROJECT_ROOT
    compiler = KernelCompiler("a2a3")
    kernel = root / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"
    print("[eager] compile-start", flush=True)
    with tempfile.TemporaryDirectory(prefix="kernel-eager-") as build_dir:
        orchestration = compiler.compile_orchestration(
            "tensormap_and_ringbuffer", str(Path(__file__).with_name("kernel_eager_orchestration.cpp")), build_dir=build_dir
        )
        incore = compiler.compile_incore(
            str(kernel), core_type="aiv", pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer"), build_dir=build_dir
        )
    print("[eager] compile-done", flush=True)
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    chip = ChipCallable.build(signature=signature, func_name="kernel_eager_orchestration", binary=orchestration, children=[(0, child)])

    lib = _load("a2a3", "onboard", "tensormap_and_ringbuffer")
    for name, argtypes in {
        "aclInit": [ctypes.c_char_p], "aclFinalize": [], "aclrtSetDevice": [ctypes.c_int],
        "aclrtResetDevice": [ctypes.c_int], "aclrtCreateStream": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtDestroyStream": [ctypes.c_void_p], "aclrtSynchronizeStreamWithTimeout": [ctypes.c_void_p, ctypes.c_int32],
        "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
        "aclrtFree": [ctypes.c_void_p], "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    }.items():
        fn = getattr(lib, name); fn.argtypes = argtypes; fn.restype = ctypes.c_int
    assert lib.aclInit(None) == 0
    print("[eager] acl-init-done", flush=True)
    stream = ctypes.c_void_p(); ctx = None; allocs = []
    try:
        assert lib.aclrtSetDevice(device) == 0
        assert lib.aclrtCreateStream(ctypes.byref(stream)) == 0
        ctx = lib.create_device_context(); assert ctx
        config = CallConfig()
        for ring in range(4):
            config.runtime_env[ring] = 64; config.runtime_env[4 + ring] = 1 << 20; config.runtime_env[8 + ring] = 1024
        aicpu, aicore, dispatcher = _binaries("a2a3", "tensormap_and_ringbuffer")
        assert lib.simpler_kernel_mode_init(ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 71) == 0
        print("[eager] kernel-init-done", flush=True)
        assert lib.simpler_kernel_mode_supported(ctx) == 1
        print("[eager] prepare-start", flush=True)
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, chip.buffer_ptr(), chip.buffer_size()) == 0
        print("[eager] prepare-done", flush=True)
        count = 128 * 128; array = ctypes.c_float * count; nbytes = ctypes.sizeof(array)
        for round_index, scalar in enumerate((1.25, -3.5)):
            src = ctypes.c_void_p(); dst = ctypes.c_void_p(); assert lib.aclrtMalloc(ctypes.byref(src), nbytes, 0) == 0; assert lib.aclrtMalloc(ctypes.byref(dst), nbytes, 0) == 0
            allocs.extend((src, dst)); values = [float(i % 127 + round_index * 257) for i in range(count)]
            host_in = array(*values); host_out = array(*([-999.0] * count))
            assert lib.aclrtMemcpy(src, nbytes, host_in, nbytes, 1) == 0
            args = ChipStorageTaskArgs(); args.add_tensor(ChipTensor.make(src.value, (count,), DataType.FLOAT32, child_memory=True)); args.add_tensor(ChipTensor.make(dst.value, (count,), DataType.FLOAT32, child_memory=True)); args.add_scalar(int.from_bytes(struct.pack("<f", scalar), "little"))
            assert lib.simpler_kernel_mode_launch(ctx, 0, args.__ptr__(), stream) == 0
            args.clear(); assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
            assert lib.aclrtMemcpy(host_out, nbytes, dst, nbytes, 2) == 0
            assert list(host_out) == [value + scalar for value in values]
        assert lib.finalize_device(ctx) == 0; ctx = None
    finally:
        if ctx: lib.finalize_device(ctx); lib.destroy_device_context(ctx)
        for ptr in reversed(allocs): lib.aclrtFree(ptr)
        if stream: lib.aclrtDestroyStream(stream)
        lib.aclrtResetDevice(device); lib.aclFinalize()


if __name__ == "__main__":
    _run_lifecycle_retry(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
