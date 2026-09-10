# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real Torch queue/allocator tests; the downstream fixture uses NPU async copy.

This is adapter evidence, not a TMR/HBG executor or binder acceptance test.
"""

import ctypes
import gc
import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.kernel import create_owner, enqueue
from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, ChipWorker

from simpler_setup.kernel_compiler import KernelCompiler


@pytest.fixture(scope="module")
def setup_adapter(request):
    if request.config.getoption("--platform") != "a2a3":
        pytest.skip("requires a2a3 and an installed Torch-NPU adapter")
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")

    adapter = importlib.import_module("_torch_npu_adapter")
    assert adapter.BUILD_TORCH_NPU_VERSION == torch_npu.__version__
    root = Path(__file__).resolve().parents[4]
    build = root / "build/kernel_torch_adapter"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(Path(__file__).parent / "native"),
            "-B",
            str(build),
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DSIMPLER_ROOT={root}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    sys.path.insert(0, str(build))
    probe = importlib.import_module("_kernel_queue_probe")
    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    torch.npu.set_device(device)
    # Known-good framework baseline before evaluating the adapter.
    assert torch.equal((torch.ones(8, device=f"npu:{device}") + 1).cpu(), torch.full((8,), 2.0))
    yield torch, probe, str(build / "libkernel_adapter_test_runtime.so"), device
    torch.npu.synchronize()


def owner_for(path, device):
    owner = create_owner(path)
    assert owner.initialize(device, b"x", b"x", b"", CallConfig(), 1) == 0
    return owner


def test_queue_keeps_inputs_and_rejects_early_close(setup_adapter):
    torch, probe, path, device = setup_adapter
    if os.environ.get("TASK_QUEUE_ENABLE") == "0":
        pytest.skip("queue hold requires asynchronous taskQueue")
    owner = owner_for(path, device)
    source = torch.arange(1024, device=f"npu:{device}", dtype=torch.float32)
    target = torch.zeros_like(source)
    expected = source.cpu()
    probe.hold()
    try:
        probe.wait_started()
        enqueue(owner, 0, [source, target], [source.numel() * source.element_size()])
        assert owner.pending == 1
        assert owner.close() != 0
        assert not owner.closed
        del source
        gc.collect()
    finally:
        probe.release()
    torch.npu.synchronize()
    assert torch.equal(target.cpu(), expected)
    assert owner.pending == 0
    assert owner.close() == 0


def test_chip_worker_uses_same_owner_for_torch_queue(setup_adapter, tmp_path):
    torch, probe, path, device = setup_adapter
    worker = ChipWorker()
    binary = tmp_path / "test.bin"
    binary.write_bytes(b"x")
    bins = SimpleNamespace(host_path=path, aicpu_path=binary, aicore_path=binary)
    worker.kernel_init(device, bins, CallConfig())
    try:
        cid = worker.kernel_prepare_callable(
            ChipCallable.build(
                signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR],
                func_name="test_copy",
                binary=b"x",
                children=[],
                scalar_count=1,
            )
        )
        source = torch.arange(64, device=f"npu:{device}", dtype=torch.float32)
        target = torch.empty_like(source)
        expected = source.cpu()
        queued = os.environ.get("TASK_QUEUE_ENABLE") != "0"
        if queued:
            probe.hold()
        try:
            if queued:
                probe.wait_started()
            enqueue(worker, cid, [source, target], [256])
            del source
            gc.collect()
            if queued:
                with pytest.raises(RuntimeError, match="-1003"):
                    worker.finalize()
                assert worker._impl.initialized
                assert cid in worker._callable_registry
        finally:
            if queued:
                probe.release()
        torch.npu.synchronize()
        assert torch.equal(target.cpu(), expected)
    finally:
        torch.npu.synchronize()
        worker.finalize()
    assert not worker._impl.initialized
    assert not worker._callable_registry


def test_many_invocations_with_storage_pressure(setup_adapter):
    torch, _, path, device = setup_adapter
    owner = owner_for(path, device)
    stream = torch.npu.Stream(device=device)
    outputs = []
    for i in range(64):
        source = torch.full((4096,), float(i), device=f"npu:{device}")
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            target = torch.empty_like(source)
            enqueue(owner, 0, [source, target], [source.numel() * source.element_size()])
            outputs.append(target)
        del source
        torch.empty(4096, device=f"npu:{device}").fill_(-99)
    torch.npu.synchronize()
    for i, target in enumerate(outputs):
        assert torch.equal(target.cpu(), torch.full((4096,), float(i)))
    assert owner.close() == 0


def test_generator_stream_and_view_offset(setup_adapter):
    torch, _, path, device = setup_adapter
    owner = owner_for(path, device)
    source = torch.arange(64, device=f"npu:{device}", dtype=torch.float32)
    target = torch.zeros(32, device=f"npu:{device}")
    original = torch.npu.current_stream()
    other = torch.npu.Stream(device=device)
    other.wait_stream(original)

    def values():
        torch.npu.set_stream(other)
        yield source[8:40]
        yield target

    try:
        enqueue(owner, 0, values(), [128])
    finally:
        torch.npu.set_stream(original)
    torch.npu.synchronize()
    assert torch.equal(target.cpu(), torch.arange(8, 40, dtype=torch.float32))
    assert owner.close() == 0


def test_tensor_rejections_do_not_admit(setup_adapter):
    torch, _, path, device = setup_adapter
    owner = owner_for(path, device)
    for tensor, message in (
        (torch.ones(1), "strided NPU storage"),
        (torch.ones((), device=f"npu:{device}"), "rank must be"),
        (torch.ones(1, device=f"npu:{device}").expand(8), "shape/stride"),
        (torch._neg_view(torch.ones(8, device=f"npu:{device}")), "unresolved negative/conjugate"),
        (torch.ones(8, device=f"npu:{device}", requires_grad=True), "inference-only"),
    ):
        with pytest.raises((RuntimeError, ValueError), match=message):
            enqueue(owner, 0, [tensor], [])
        assert owner.pending == 0
    assert owner.close() == 0


def test_uncached_storage_is_rejected(setup_adapter):
    if os.environ.get("PYTORCH_NO_NPU_MEMORY_CACHING") != "1":
        pytest.skip("run separately with caching disabled before importing Torch-NPU")
    torch, _, path, device = setup_adapter
    owner = owner_for(path, device)
    try:
        tensor = torch.ones(8, device=f"npu:{device}")
        with pytest.raises(RuntimeError, match="invalid device pointer"):
            enqueue(owner, 0, [tensor], [])
        assert owner.pending == 0
    finally:
        assert owner.close() == 0


@pytest.mark.parametrize("kind", ["same_storage", "different_allocations", "empty"])
def test_real_record_stream_counts(setup_adapter, kind):
    torch, probe, path, device = setup_adapter
    owner = owner_for(path, device)
    runtime = ctypes.CDLL(path)
    runtime.test_launch_count.argtypes = []
    runtime.test_launch_count.restype = ctypes.c_int
    if kind == "same_storage":
        backing = torch.arange(32, device=f"npu:{device}", dtype=torch.float32)
        source, target = backing[:8], backing[16:24]
        expected_counts = {backing.untyped_storage().data_ptr(): 1}
    elif kind == "different_allocations":
        # Distinct blocks in one caching segment must not be deduplicated.
        candidates = [torch.full((8,), float(i), device=f"npu:{device}") for i in range(16)]
        segments = {}
        for tensor in candidates:
            segment = probe.allocation_base(tensor.untyped_storage().data_ptr())
            if segment in segments:
                source, target = segments[segment], tensor
                break
            segments[segment] = tensor
        else:
            pytest.fail("could not construct two live allocations in the same caching segment")
        expected_counts = {source.untyped_storage().data_ptr(): 1, target.untyped_storage().data_ptr(): 1}
        assert len(expected_counts) == 2
    else:
        source = torch.empty((0,), device=f"npu:{device}")
        target = torch.empty_like(source)
        assert source.untyped_storage().data_ptr() == target.untyped_storage().data_ptr() == 0
        expected_counts = {0: 0}
    expected = source.cpu().clone()
    torch.npu.synchronize()
    launches = runtime.test_launch_count()
    probe.start_recording()
    try:
        enqueue(owner, 0, [source, target], [source.numel() * source.element_size()])
        torch.npu.synchronize()
        assert runtime.test_launch_count() == launches + 1
        assert owner.pending == 0
        assert {addr: probe.record_count(addr) for addr in expected_counts} == expected_counts
    finally:
        torch.npu.synchronize()
        probe.stop_recording()
        assert owner.close() == 0
    assert torch.equal(target.cpu(), expected)


def test_recognizable_external_storage_rejected(setup_adapter):
    torch, probe, path, device = setup_adapter
    owner = owner_for(path, device)
    source = torch.ones(8, device=f"npu:{device}")
    foreign = probe.external_alias(source)
    assert foreign.data_ptr() == source.data_ptr()
    try:
        with pytest.raises(ValueError, match="cannot retain external Tensor storage"):
            enqueue(owner, 0, [foreign], [])
        assert owner.pending == 0
    finally:
        assert owner.close() == 0


def test_framework_logical_slot_requires_caller_lease(setup_adapter):
    torch, probe, path, device = setup_adapter
    if os.environ.get("TASK_QUEUE_ENABLE") == "0":
        pytest.skip("pending window requires asynchronous taskQueue")

    class Slot:
        def __init__(self):
            self.tensor = torch.arange(32, device=f"npu:{device}", dtype=torch.float32)
            self.leased = False

        def acquire(self):
            assert not self.leased
            self.leased = True
            return self.tensor[8:16]

        def recycle(self):
            if self.leased:
                raise RuntimeError("framework slot is still leased")
            self.tensor.fill_(-1)

    slot = Slot()
    owner = owner_for(path, device)
    source = slot.acquire()
    target = torch.empty_like(source)
    torch.npu.synchronize()
    probe.hold()
    try:
        probe.wait_started()
        enqueue(owner, 0, [source, target], [32])
        del source
        assert owner.pending == 1
        # K8 can retain Torch storage, but cannot infer this logical slot lease.
        with pytest.raises(RuntimeError, match="still leased"):
            slot.recycle()
    finally:
        probe.release()
    torch.npu.synchronize()
    assert torch.equal(target.cpu(), torch.arange(8, 16, dtype=torch.float32))
    slot.leased = False
    slot.recycle()
    torch.npu.synchronize()
    assert owner.close() == 0


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_public_runtime_admission(setup_adapter, runtime, tmp_path, capfd, st_platform):
    """Real runtime admission, not a substitute for the missing executor/binder."""
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        pytest.skip("exercise the expected native error inline, without poisoning the SDK queue")
    torch, _, _, device = setup_adapter
    root = Path(__file__).resolve().parents[4]
    base = root / "build/lib" / st_platform / "onboard" / runtime
    owner = create_owner(str(base / "libhost_runtime.so"))
    try:
        rc = owner.initialize(
            device,
            (base / "libaicpu_kernel.so").read_bytes(),
            (base / "aicore_kernel.o").read_bytes(),
            (root / "build/lib" / st_platform / "dispatcher/libsimpler_aicpu_dispatcher.so").read_bytes(),
            CallConfig(),
            1,
        )
        if runtime == "host_build_graph":
            assert rc == -1001  # This stack has no HBG kernel resource builder.
            return
        assert rc == 0
        assert owner.supported() == 0
        binary = KernelCompiler(st_platform).compile_orchestration(
            runtime,
            str(root / "tests/ut/py/kernel_prepare_orchestration.cpp"),
            build_dir=str(tmp_path),
        )
        chip = ChipCallable.build(signature=[], func_name="kernel_prepare_orchestration", binary=binary, children=[])
        image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
        assert owner.prepare(0, image) == 0
        with pytest.raises(RuntimeError):
            enqueue(owner, 0, [], [])
        assert "kernel native enqueue failed: status=-1003" in capfd.readouterr().err
        assert owner.pending == 0
    finally:
        torch.npu.synchronize()
        assert owner.close() == 0
