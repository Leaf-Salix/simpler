# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Communication-domain identity and inherited-use integration on real sim workers."""

from __future__ import annotations

import ctypes
import os
import select
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import (
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    RemoteTensorRef,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_NUMEL = 128 * 128
_NBYTES = _NUMEL * 4


def _chip_callable(platform):
    compiler = KernelCompiler(platform=platform)
    source = Path(__file__).parent / "kernels/orchestration/domain_tensor_orch.cpp"
    binary = compiler.compile_orchestration(runtime_name=_RUNTIME, source_path=str(source))
    incore_source = Path(__file__).parents[1] / "comm_region/recursive_single_owner/kernels/aiv/kernel_transform.cpp"
    incore = compiler.compile_incore(
        str(incore_source),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(_RUNTIME),
    )
    if not platform.endswith("sim"):
        incore = extract_text_section(incore)
    return ChipCallable.build(
        signature=[D.IN, D.OUT],
        func_name="domain_tensor_orch",
        binary=binary,
        children=[(0, CoreCallable.build(signature=[D.IN, D.OUT, D.SCALAR], binary=incore))],
    )


def _worker(platform, device_ids):
    return Worker(level=3, platform=platform, runtime=_RUNTIME, device_ids=device_ids, num_sub_workers=0)


@pytest.mark.platforms(["a2a3sim", "a5sim", "a2a3", "a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_reuse_preserves_data_across_runs(st_platform, st_device_ids):
    worker = _worker(st_platform, [int(st_device_ids[0])])
    saved = []
    spec = CommBufferSpec("payload", "int32", 1, 4)
    try:
        worker.init()

        def first(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(
                name="tp", members=[0], window_size=4096, buffers=[spec], retain_after_run=True
            )
            orch.copy_to_domain(domain, 0, struct.pack("<i", 123), buffer="payload")
            saved.append(domain)

        worker.run(first)

        def second(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(
                name="tp", members=[0], window_size=4096, buffers=[spec], retain_after_run=True
            )
            assert domain is saved[0]
            assert orch.get_domain("tp") is domain
            assert orch.get_domain(domain.ref) is domain
            assert orch.copy_from_domain(domain, 0, 4, buffer="payload") == struct.pack("<i", 123)
            domain.release()

        worker.run(second)
        assert saved[0].freed
        assert worker.live_domains() == {}
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a5sim", "a2a3", "a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_completed_wait_in_callback_rebuilds_domain_and_computes_view(st_platform, st_device_ids):
    worker = _worker(st_platform, [int(st_device_ids[0])])
    chip = worker.register(_chip_callable(st_platform))
    saved = []
    spec = CommBufferSpec("payload", "float32", _NUMEL + 16, _NBYTES + 64)
    try:
        worker.init()
        output = worker.create_buffer(_NBYTES)

        def first(orch, _args, _cfg):
            saved.append(
                orch.allocate_domain(
                    name="tp",
                    members=[0],
                    window_size=spec.nbytes,
                    buffers=[spec],
                    retain_after_run=True,
                )
            )

        prior = worker.submit(first)
        prior.wait()
        original_identity = saved[0].member(0).buffers["payload"].identity

        def second(orch, _args, config):
            saved[0].release()
            prior.wait()
            assert saved[0].released and not saved[0].freed
            domain = orch.allocate_domain(name="tp", members=[0], window_size=spec.nbytes, buffers=[spec])
            assert domain.ref != saved[0].ref
            payload = domain.member(0).buffers["payload"]
            assert payload.identity != original_identity
            orch.copy_to_domain(
                domain,
                0,
                struct.pack(f"<{_NUMEL + 16}f", *([-1.0] * 16 + [35.0] * _NUMEL)),
                buffer="payload",
            )
            view = payload.tensor([_NUMEL], DataType.FLOAT32, byte_offset=64)
            assert view.buffer.identity == payload.tensor([16], DataType.FLOAT32).buffer.identity
            args = TaskArgs()
            args.add_tensor(view, TensorArgType.INPUT)
            args.add_tensor(output.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(chip, args, config, worker=0)
            domain.release()
            saved.append(domain)

        worker.run(second)
        assert struct.unpack_from(f"<{_NUMEL}f", output.shm.buf) == (42.0,) * _NUMEL
        assert all(domain.released and domain.freed for domain in saved)
        assert worker.live_domains() == {} and worker._domain_registry.records == {}

        def reject_old_refs(orch, _args, _cfg):
            for domain in saved:
                with pytest.raises(KeyError):
                    orch.get_domain(domain.ref)

        worker.run(reject_old_refs)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a5sim", "a2a3", "a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_l3_domain_tensor_reaches_chip_after_release(st_platform, st_device_ids):
    worker = _worker(st_platform, [int(st_device_ids[0])])
    chip = worker.register(_chip_callable(st_platform))
    saved = []
    try:
        worker.init()
        output = worker.create_buffer(_NBYTES)

        def build(orch, _args, config):
            domain = orch.allocate_domain(
                name="chip",
                members=[0],
                window_size=_NBYTES,
                buffers=[CommBufferSpec("payload", "float32", _NUMEL, _NBYTES)],
            )
            orch.copy_to_domain(domain, 0, struct.pack(f"<{_NUMEL}f", *([35.0] * _NUMEL)), buffer="payload")
            args = TaskArgs()
            args.add_tensor(domain.member(0).buffers["payload"].tensor([_NUMEL], DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(output.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(chip, args, config, worker=0)
            domain.release()
            saved.append(domain)

        worker.run(build)
        assert struct.unpack_from(f"<{_NUMEL}f", output.shm.buf) == (42.0,) * _NUMEL
        assert saved[0].released and saved[0].freed
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("ref_only", [False, True])
def test_l4_tensor_reaches_chip_after_parent_release(st_platform, st_device_ids, ref_only):
    l3 = _worker(st_platform, [int(st_device_ids[0])])
    parent = Worker(level=4, platform=st_platform, runtime=_RUNTIME, num_sub_workers=0)
    gate_read, gate_write = os.pipe()
    callable_obj = _chip_callable(st_platform)
    chip = l3.register(callable_obj)
    node = parent.add_worker(l3)
    parent.register(callable_obj)

    def forward(orch, args, config):
        assert select.select([gate_read], [], [], 10)[0], "parent did not release the accepted-task gate"
        assert os.read(gate_read, 1) == b"1"
        domain = orch.get_domain(args.domain_refs()[0])
        assert domain.borrowed
        if not ref_only:
            assert domain.member(0).buffers["payload"].to_descriptor() == args.tensor(0).buffer
        assert domain.local_context(0).device_ctx
        with pytest.raises(RuntimeError, match="borrowed"):
            domain.release()
        chip_args = TaskArgs()
        source = domain.member(0).buffers["payload"].tensor([_NUMEL], DataType.FLOAT32) if ref_only else args.tensor(0)
        chip_args.add_tensor(source, TensorArgType.INPUT)
        chip_args.add_tensor(args.tensor(0 if ref_only else 1), TensorArgType.OUTPUT_EXISTING)
        orch.submit_next_level(chip, chip_args, config, worker=0)

    child = parent.register(forward)
    saved = []
    try:
        parent.init()
        output = parent.create_buffer(_NBYTES)

        def build(orch, _args, config):
            domain = orch.allocate_domain(
                name="tp",
                members=[(node, 0)],
                window_size=_NBYTES,
                buffers=[CommBufferSpec("payload", "float32", _NUMEL, _NBYTES)],
            )
            orch.copy_to_domain(domain, 0, struct.pack(f"<{_NUMEL}f", *([35.0] * _NUMEL)), buffer="payload")
            args = TaskArgs()
            args.add_domain_ref(domain.ref)
            if not ref_only:
                args.add_tensor(
                    domain.member(0).buffers["payload"].tensor([_NUMEL], DataType.FLOAT32), TensorArgType.INPUT
                )
            args.add_tensor(output.tensor([_NUMEL], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            orch.submit_next_level(child, args, config, worker=node)
            domain.release()
            os.write(gate_write, b"1")
            saved.append(domain)

        parent.run(build)
        assert struct.unpack_from(f"<{_NUMEL}f", output.shm.buf) == (42.0,) * _NUMEL
        assert saved[0].released and saved[0].freed
    finally:
        try:
            os.write(gate_write, b"1")
            parent.close()
            l3.close()
        finally:
            os.close(gate_read)
            os.close(gate_write)


def remote_forward_domain_tensor(orch, args, config):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    domain = orch.get_domain(args.domain_refs()[0])
    assert domain.borrowed
    assert domain.member(0).buffers["payload"].to_descriptor() == args.tensor(0).buffer
    with socket.create_connection(("127.0.0.1", int(args.scalar(0))), timeout=10) as connection:
        assert connection.recv(1) == b"1"
    digest = b"".join(int(args.scalar(index)).to_bytes(8, "little") for index in range(1, 5))
    chip_args = TaskArgs()
    chip_args.add_tensor(args.tensor(0), TensorArgType.INPUT)
    chip_args.add_tensor(args.tensor(1), TensorArgType.OUTPUT_EXISTING)
    orch.submit_next_level(get_inner_handle(digest.hex()), chip_args, config, worker=0)


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_tcp_domain_tensor_and_remote_buffer_after_release(st_platform):
    from tests.ut.py.test_global_comm_domain import _free_tcp_port, _wait_for_tcp_ports  # noqa: PLC0415

    daemon_port = _free_tcp_port()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(daemon_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    parent = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=20)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(10)
    release_gate = threading.Event()
    gate_errors = []

    def serve_gate():
        try:
            assert release_gate.wait(10)
            connection, _ = listener.accept()
            with connection:
                connection.sendall(b"1")
        except BaseException as exc:
            gate_errors.append(exc)

    gate_thread = None
    captured = []
    try:
        _wait_for_tcp_ports((daemon_port,))
        node = parent.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=f"127.0.0.1:{daemon_port}", platform=st_platform, device_ids=(0,), comm_profile="sim"
            )
        )
        chip = parent.register(_chip_callable(st_platform))
        remote = parent.register(
            RemoteCallable("tests.st.worker.comm_domain.test_unified_domain:remote_forward_domain_tensor"),
            workers=[node],
        )
        parent.init()
        output = parent.remote_malloc(worker=node, nbytes=_NBYTES)
        gate_thread = threading.Thread(target=serve_gate, daemon=True)
        gate_thread.start()

        def build(orch, _args, config):
            domain = orch.allocate_domain(
                name="tcp",
                members=[(node, 0)],
                window_size=_NBYTES,
                buffers=[CommBufferSpec("payload", "float32", _NUMEL, _NBYTES)],
            )
            orch.copy_to_domain(domain, 0, struct.pack(f"<{_NUMEL}f", *([35.0] * _NUMEL)), buffer="payload")
            args = TaskArgs()
            args.add_tensor(domain.member(0).buffers["payload"].tensor([_NUMEL], DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(
                RemoteTensorRef(output, shape=(_NUMEL,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING
            )
            args.add_scalar(listener.getsockname()[1])
            for offset in range(0, 32, 8):
                args.add_scalar(int.from_bytes(chip.digest[offset : offset + 8], "little"))
            orch.submit_next_level(remote, args, config, worker=node)
            domain.release()
            release_gate.set()
            captured.append(domain)

        parent.run(build)
        result = (ctypes.c_ubyte * _NBYTES)()
        parent.remote_copy_from(output, result, _NBYTES)
        assert struct.unpack(f"<{_NUMEL}f", result) == (42.0,) * _NUMEL
        assert captured[0].freed
        gate_thread.join(10)
        assert not gate_thread.is_alive() and not gate_errors
        parent.remote_free(output)
    finally:
        release_gate.set()
        listener.close()
        if gate_thread is not None:
            gate_thread.join(10)
        try:
            parent.close()
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)
