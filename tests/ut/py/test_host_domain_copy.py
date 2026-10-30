# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host staging shares the existing Run control-ordering contract."""

import ctypes
import time
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.task_interface import HostWindowSpec
from simpler.worker import Worker


@pytest.fixture
def host_domain():
    worker = Worker(
        level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=1
    )
    worker.init()
    saved = []

    def allocate(orch, _args, _cfg):
        domain = orch.get_or_allocate_domain(
            name="table", members=[0, 1], window_size=4096, host_window=HostWindowSpec(128), retain_after_run=True
        )
        orch.copy_to_host_domain(domain, b"original", offset=16)
        saved.append(domain)

    try:
        worker.run(allocate)
        yield worker, saved[0]
    finally:
        worker.close()


def test_host_copy_outside_callback_refuses_in_flight_run_without_rewriting(host_domain):
    worker, domain = host_domain
    signal = SharedMemory(create=True, size=1)
    assert signal.buf is not None
    signal.buf[0] = 0
    name = signal.name

    def pending(_args):
        gate = SharedMemory(name=name)
        assert gate.buf is not None
        try:
            deadline = time.monotonic() + 10
            while gate.buf[0] == 0:
                if time.monotonic() >= deadline:
                    raise TimeoutError("test did not release pending task")
        finally:
            gate.close()

    run = None
    try:
        target = worker.register(pending)
        run = worker.submit(lambda orch, _args, _cfg: orch.submit_sub(target))
        assert not run.done
        with pytest.raises(RuntimeError, match="still in flight"):
            worker._orch.copy_to_host_domain(domain, b"modified", offset=16)
        with pytest.raises(RuntimeError, match="still in flight"):
            worker._orch.copy_from_host_domain(domain, 8, offset=16)
        assert ctypes.string_at(worker._domain_state(domain.ref).host_base + 16, 8) == b"original"
    finally:
        assert signal.buf is not None
        signal.buf[0] = 1
        try:
            if run is not None:
                run.wait(10)
        finally:
            signal.close()
            signal.unlink()
    worker._orch.copy_to_host_domain(domain, b"modified", offset=16)
    assert worker._orch.copy_from_host_domain(domain, 8, offset=16) == b"modified"


def test_host_copy_after_callback_submission_refuses_without_rewriting(host_domain):
    worker, domain = host_domain
    target = worker.register(lambda _args: None)

    def submit_then_update(orch, _args, _cfg):
        orch.copy_to_host_domain(domain, b"prepared", offset=16)
        orch.submit_sub(target)
        with pytest.raises(RuntimeError, match="cannot follow a task submission"):
            orch.copy_to_host_domain(domain, b"modified", offset=16)
        with pytest.raises(RuntimeError, match="cannot follow a task submission"):
            orch.copy_from_host_domain(domain, 8, offset=16)
        assert ctypes.string_at(worker._domain_state(domain.ref).host_base + 16, 8) == b"prepared"

    worker.run(submit_then_update)
    assert worker._orch.copy_from_host_domain(domain, 8, offset=16) == b"prepared"


def test_host_copy_is_explicit_and_bounded(host_domain):
    worker, domain = host_domain
    orch = worker._orch
    with pytest.raises(TypeError):
        orch.copy_to_domain(domain, None, b"modified", offset=16)
    with pytest.raises(TypeError):
        orch.copy_from_domain(domain, None, 8, offset=16)
    with pytest.raises(ValueError, match="exceeds"):
        orch.copy_to_host_domain(domain, b"modified", offset=124)
    assert orch.copy_from_host_domain(domain, 8, offset=16) == b"original"
