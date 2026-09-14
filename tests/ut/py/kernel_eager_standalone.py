# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import os
import struct
import tempfile
from pathlib import Path

from simpler.task_interface import ArgDirection, ChipCallable, ChipStorageTaskArgs, ChipTensor, CoreCallable, DataType
from test_kernel_mode_c_api import _PROJECT_ROOT, CallConfig, _load

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

print("compile", flush=True)
with tempfile.TemporaryDirectory(prefix="kernel-eager-") as build_dir:
    compiler = KernelCompiler("a2a3")
    orchestration = compiler.compile_orchestration(
        "tensormap_and_ringbuffer", str(Path(__file__).with_name("kernel_eager_orchestration.cpp")), build_dir=build_dir
    )
    incore = compiler.compile_incore(
        str(_PROJECT_ROOT / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer"),
        build_dir=build_dir,
    )
child = CoreCallable.build(
    signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR], binary=extract_text_section(incore)
)
chip = ChipCallable.build(
    signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR],
    func_name="kernel_eager_orchestration",
    binary=orchestration,
    children=[(0, child)],
)
lib = _load("a2a3", "onboard", "tensormap_and_ringbuffer")
for name, argtypes in {
    "aclInit": [ctypes.c_char_p],
    "aclFinalize": [],
    "aclrtSetDevice": [ctypes.c_int],
    "aclrtResetDevice": [ctypes.c_int],
    "aclrtCreateStream": [ctypes.POINTER(ctypes.c_void_p)],
    "aclrtDestroyStream": [ctypes.c_void_p],
    "aclrtSynchronizeStreamWithTimeout": [ctypes.c_void_p, ctypes.c_int32],
    "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
    "aclrtFree": [ctypes.c_void_p],
    "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
}.items():
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
print("aclInit", lib.aclInit(None), flush=True)
stream = ctypes.c_void_p()
ctx = None
allocs = []
try:
    device = 0
    print("set/create", lib.aclrtSetDevice(device), lib.aclrtCreateStream(ctypes.byref(stream)), flush=True)
    ctx = lib.create_device_context()
    config = CallConfig()
    for ring in range(4):
        config.runtime_env[ring] = 64
        config.runtime_env[4 + ring] = 1 << 20
        config.runtime_env[8 + ring] = 1024
    base = _PROJECT_ROOT / "build" / "lib" / "a2a3" / "onboard" / "tensormap_and_ringbuffer"
    dispatcher_path = _PROJECT_ROOT / "build" / "lib" / "a2a3" / "dispatcher" / "libsimpler_aicpu_dispatcher.so"
    aicpu, aicore, dispatcher = (
        (base / "libaicpu_kernel.so").read_bytes(),
        (base / "aicore_kernel_mode.o").read_bytes(),
        dispatcher_path.read_bytes(),
    )
    print(
        "init",
        lib.simpler_kernel_mode_init(
            ctx, device, aicpu, len(aicpu), aicore, len(aicore), dispatcher, len(dispatcher), ctypes.byref(config), 71
        ),
        flush=True,
    )
    print("supported", lib.simpler_kernel_mode_supported(ctx), flush=True)
    prepare_rc = lib.simpler_kernel_mode_prepare_callable(ctx, 0, chip.buffer_ptr(), chip.buffer_size())
    print("prepare", prepare_rc, flush=True)
    if prepare_rc != 0:
        raise RuntimeError(f"kernel callable preparation failed: {prepare_rc}")
    count = 128 * 128
    array = ctypes.c_float * count
    nbytes = ctypes.sizeof(array)
    rounds = int(os.environ.get("SIMPLER_EAGER_ROUNDS", "1"))
    if rounds < 1 or rounds > 2:
        raise ValueError("SIMPLER_EAGER_ROUNDS must be 1 or 2")
    for round_index, scalar in enumerate((1.25, -3.5)[:rounds]):
        src = ctypes.c_void_p()
        dst = ctypes.c_void_p()
        print(
            "malloc",
            lib.aclrtMalloc(ctypes.byref(src), nbytes, 0),
            lib.aclrtMalloc(ctypes.byref(dst), nbytes, 0),
            flush=True,
        )
        allocs.extend((src, dst))
        values = [float(i % 127 + round_index * 257) for i in range(count)]
        host_in = array(*values)
        host_out = array(*([-999.0] * count))
        print("h2d", lib.aclrtMemcpy(src, nbytes, host_in, nbytes, 1), flush=True)
        args = ChipStorageTaskArgs()
        args.add_tensor(ChipTensor.make(src.value, (count,), DataType.FLOAT32, child_memory=True))
        args.add_tensor(ChipTensor.make(dst.value, (count,), DataType.FLOAT32, child_memory=True))
        args.add_scalar(int.from_bytes(struct.pack("<f", scalar), "little"))
        launch_rc = lib.simpler_kernel_mode_launch(ctx, 0, args.__ptr__(), stream)
        print("launch", launch_rc, flush=True)
        args.clear()
        if launch_rc != 0:
            raise RuntimeError(f"kernel eager launch failed: {launch_rc}")
        sync_rc = lib.aclrtSynchronizeStreamWithTimeout(stream, 30000)
        print("sync", sync_rc, flush=True)
        if sync_rc != 0:
            raise RuntimeError(f"eager stream synchronization failed: {sync_rc}")
        print(
            "d2h",
            lib.aclrtMemcpy(host_out, nbytes, dst, nbytes, 2),
            "first",
            host_out[0],
            "expect",
            values[0] + scalar,
            flush=True,
        )
    print("finalize", lib.finalize_device(ctx), flush=True)
    ctx = None
finally:
    if ctx:
        print("finalize-finally", lib.finalize_device(ctx), flush=True)
    if ctx:
        lib.destroy_device_context(ctx)
    for ptr in reversed(allocs):
        lib.aclrtFree(ptr)
    if stream:
        lib.aclrtDestroyStream(stream)
    print("reset/final", lib.aclrtResetDevice(0), lib.aclFinalize(), flush=True)
