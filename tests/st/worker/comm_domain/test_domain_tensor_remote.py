# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A creator's logical domain Tensor reaches a TCP L3 and its exact chip."""

import socket
import struct
import subprocess
import sys
import time

import pytest
from simpler.task_interface import CommBufferSpec, CommDomainRef, DataType, TaskArgs, TensorArgType
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from examples.workers.l3.multi_chip_dispatch.main import N_COLS, N_ROWS, build_chip_callable

_NUMEL = N_ROWS * N_COLS
_NBYTES = _NUMEL * 4
_RUNTIME = "tensormap_and_ringbuffer"


def remote_domain_add(orch, args, cfg):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    ref = CommDomainRef.from_tensor(args.tensor(0))
    domain = orch.get_domain(ref)
    assert domain.borrowed and domain.ref == ref
    payload = domain.member(0).buffers["payload"]
    assert payload.identity == args.tensor(0).buffer.identity == args.tensor(1).buffer.identity
    assert domain.buffers[0].count == _NUMEL + 32
    assert orch.copy_from_domain(domain, 0, 4, buffer="payload", offset=64) == struct.pack("<f", 35.0)
    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little") for i in range(4))
    chip = get_inner_handle(digest.hex())
    chip_args = TaskArgs()
    for i, tag in enumerate((TensorArgType.INPUT, TensorArgType.INPUT, TensorArgType.OUTPUT_EXISTING)):
        chip_args.add_tensor(args.tensor(i), tag)
    with pytest.raises(ValueError, match="wrong worker|stale"):
        orch.submit_next_level(chip, chip_args, cfg, worker=1)
    orch.submit_next_level(chip, chip_args, cfg, worker=0)


@pytest.fixture
def tcp_daemon(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (tmp_path / "daemon.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    if time.monotonic() >= deadline or process.poll() is not None:
                        raise RuntimeError("TCP daemon did not start")
                    time.sleep(0.01)
            yield f"127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.platforms(["a2a3sim", "a5sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_domain_tensor_reaches_remote_chip(tcp_daemon, st_platform, st_device_ids):
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=30)
    try:
        node = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=tcp_daemon,
                platform=st_platform,
                device_ids=(int(st_device_ids[0]),),
                comm_profile="sim",
            )
        )
        chip = worker.register(build_chip_callable(st_platform))
        remote = worker.register(RemoteCallable(__name__ + ":remote_domain_add"), workers=[node])
        worker.init()
        request = dict(
            name="tp",
            members=[(node, 0)],
            window_size=2 * _NBYTES + 4096,
            buffers=[
                CommBufferSpec("payload", "float32", _NUMEL + 32, _NBYTES + 128),
                CommBufferSpec("result", "float32", _NUMEL, _NBYTES),
            ],
            retain_after_run=True,
        )
        saved = []

        def first(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(**request)
            orch.copy_to_domain(domain, 0, struct.pack(f"<{_NUMEL + 32}f", *([35.0] * (_NUMEL + 32))), buffer="payload")
            saved.append(domain)

        worker.run(first)
        allocation_count = worker._next_alloc_id

        def second(orch, _args, cfg):
            domain = orch.get_or_allocate_domain(**request)
            assert domain is saved[0] and worker._next_alloc_id == allocation_count
            buffers = domain.member(0).buffers
            args = TaskArgs()
            args.add_tensor(buffers["payload"].tensor([_NUMEL], DataType.FLOAT32, byte_offset=64), TensorArgType.INPUT)
            args.add_tensor(buffers["payload"].tensor([_NUMEL], DataType.FLOAT32, byte_offset=128), TensorArgType.INPUT)
            args.add_tensor(buffers["result"].tensor([_NUMEL], DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            for i in range(0, 32, 8):
                args.add_scalar(int.from_bytes(chip.digest[i : i + 8], "little"))
            orch.submit_next_level(remote, args, cfg, worker=node)

        worker.run(second)

        def finish(orch, _args, _cfg):
            domain = orch.get_domain(saved[0].ref)
            assert (
                struct.unpack(f"<{_NUMEL}f", orch.copy_from_domain(domain, 0, _NBYTES, buffer="result"))
                == (70.0,) * _NUMEL
            )
            domain.release()

        worker.run(finish)
        assert saved[0].freed and worker.live_domains == {}
    finally:
        worker.close()
