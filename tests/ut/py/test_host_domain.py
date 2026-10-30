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
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from simpler import host_domain_backend
from simpler.buffer import AddressSpace, ImportContext, ImportRegistry
from simpler.comm_endpoints import DEVICE_AICPU, AdapterProfile
from simpler.global_comm_domain import GLOBAL_DOMAIN_PROFILE_A3_FABRIC, GLOBAL_DOMAIN_PROFILE_IDS
from simpler.host_domain import HostDomainResource
from simpler.host_domain_backend import A3HostBacking, host_attachment_profile
from simpler.task_interface import CommBufferSpec, DataType, HostWindowSpec, TaskArgs, TensorTransfer
from simpler.worker import Worker


@pytest.fixture
def host_worker():
    worker = Worker(
        level=3, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_ids=[0, 1], num_sub_workers=0
    )
    worker.init()
    try:
        yield worker
    finally:
        worker.close()


def request(**changes):
    result = dict(
        name="host",
        members=[1, 0],
        window_size=4096,
        buffers=[CommBufferSpec("output", "float32", 16, 64)],
        host_window=HostWindowSpec(128),
        retain_after_run=True,
    )
    result.update(changes)
    return result


def allocate(worker, **changes):
    saved = []
    worker.run(lambda orch, _args, _cfg: saved.append(orch.get_or_allocate_domain(**request(**changes))))
    return saved[0]


def test_host_domain_preserves_native_identity_and_target_views(host_worker):
    domain = allocate(host_worker)
    source = domain.host().tensor((16,), DataType.FLOAT32)
    views = [domain.map_view(source, domain_rank=rank) for rank in range(2)]
    assert source.mapping_ref == 0
    assert source.buffer.address_space == AddressSpace.HOST
    assert all(view.buffer == source.buffer for view in views)
    assert [view.mapping_ref for view in views] == [1, 2]
    for rank, view in enumerate(views):
        args = TaskArgs()
        args.add_tensor(view, transfer=TensorTransfer.NONE)
        with host_worker._child_prov_lock:
            host_worker._child_prov_check_dispatch_locked(args, 1 - rank, api="test")
            with pytest.raises(ValueError, match="target worker"):
                host_worker._child_prov_check_dispatch_locked(args, rank, api="test")
    native_args = TaskArgs()
    native_args.add_tensor(source)
    with host_worker._child_prov_lock, pytest.raises(ValueError, match="explicit mapping"):
        host_worker._child_prov_check_dispatch_locked(native_args, 0, api="test")
    domain.release()
    assert domain.freed
    with pytest.raises(RuntimeError, match="released"):
        domain.map_view(source, domain_rank=0)


def test_host_domain_cpu_copy_and_configuration_reuse(host_worker):
    saved = []
    data = struct.pack("<16f", *range(16))

    def fill(orch, _args, _cfg):
        domain = orch.get_or_allocate_domain(**request())
        orch.copy_to_domain(domain, None, data, offset=64)
        assert orch.copy_from_domain(domain, None, len(data), offset=64) == data
        assert orch.get_or_allocate_domain(**request()) is domain
        with pytest.raises(ValueError, match="configuration differs"):
            orch.get_or_allocate_domain(**request(host_window=HostWindowSpec(256)))
        with pytest.raises(ValueError, match="exceeds"):
            orch.copy_to_domain(domain, None, data, offset=68)
        saved.append(domain)

    host_worker.run(fill)
    assert not saved[0].freed
    saved[0].release()
    assert saved[0].freed


def test_host_domain_stale_tensor_cannot_rebind_to_same_name(host_worker):
    domain = allocate(host_worker)
    stale = domain.map_view(domain.host().tensor((16,), DataType.FLOAT32), domain_rank=0)
    domain.release()
    replacement = allocate(host_worker)
    assert replacement.host().identity != stale.buffer.identity
    args = TaskArgs()
    args.add_tensor(stale, transfer=TensorTransfer.NONE)
    with host_worker._child_prov_lock, pytest.raises(ValueError, match="not a live"):
        host_worker._child_prov_check_dispatch_locked(args, 1, api="test")
    replacement.release()


def test_host_domain_native_reference_requires_endpoint_grant(host_worker):
    domain = allocate(host_worker)
    tensor = domain.host().tensor((16,), DataType.FLOAT32)
    registry = ImportRegistry(ImportContext(DEVICE_AICPU))
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(tensor)
    domain.release()


def test_shared_host_backing_keeps_logical_bounds_and_endpoint_profile(host_worker):
    domain = allocate(host_worker)
    resource = domain._host_resource
    assert domain.host().nbytes == 128
    assert resource.export.mapping_size % os.sysconf("SC_PAGE_SIZE") == 0
    assert resource.export.mapping_size >= domain.host().nbytes
    assert host_attachment_profile(resource.export, "a2a3sim") == AdapterProfile.HOST_SHM_MAP
    with pytest.raises(ValueError, match="does not match"):
        host_attachment_profile(resource.export, "a2a3")
    with pytest.raises(ValueError, match="exceeds"):
        resource.copy(128, b"x", writing=True)
    domain.release()


def test_fabric_backing_preserves_logical_extent_and_authorizes_actual_pids(monkeypatch):
    calls = []

    class Native:
        base = 4096

        def allocate(self, device, numa, size):
            calls.append((device, numa, size))
            return 2097152, bytes(128)

        def authorize(self, pids):
            calls.append(tuple(pids))

        def release(self):
            calls.append("release")

    monkeypatch.setattr(host_domain_backend, "_HostDomainMemory", Native)
    backing = A3HostBacking()
    base, export = backing.allocate(HostWindowSpec(128, numa_id=1), None, 2, 2)
    assert base == 4096
    assert export.mapping_size == 2097152
    assert export.profile_id == GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC]
    assert host_attachment_profile(export, "a2a3") == AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT
    with pytest.raises(ValueError, match="does not match"):
        host_attachment_profile(export, "a2a3sim")
    backing.authorize([31, 42])
    backing.release()
    assert calls == [(2, 1, 128), (31, 42), "release"]


@pytest.mark.parametrize("kwargs", [dict(nbytes=0), dict(nbytes=63), dict(nbytes=True), dict(nbytes=64, numa_id=-1)])
def test_host_window_spec_rejects_invalid_allocation(kwargs):
    with pytest.raises((ValueError, TypeError)):
        HostWindowSpec(**kwargs)


def test_owner_cleanup_retries_without_replaying_chip_release(host_worker, monkeypatch):
    domain = allocate(host_worker)
    resource = domain._host_resource
    original = resource.backend.release
    attempts = []

    def fail_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("injected Host owner free failure")
        original()

    monkeypatch.setattr(resource.backend, "release", fail_once)
    with pytest.raises(RuntimeError, match="injected Host owner"):
        domain.release()
    assert resource.views_released
    assert host_worker._domain_free_results[domain.allocation_id] is None

    def no_replay(**_kwargs):
        raise AssertionError("confirmed chip collective must not replay")

    monkeypatch.setattr(host_worker, "_dispatch_control_domain", no_replay)
    host_worker.close()
    assert len(attempts) == 2
    assert domain.freed and host_worker._cleanup_journal.empty


def test_failed_creation_without_any_import_releases_owner(host_worker, monkeypatch):
    captured = []
    original = HostDomainResource.allocate

    def capture(resource):
        original(resource)
        captured.append(resource)

    monkeypatch.setattr(HostDomainResource, "allocate", capture)

    def refuse(**_kwargs):
        raise RuntimeError("all children refused before allocation")

    monkeypatch.setattr(host_worker, "_dispatch_control_domain", refuse)
    with pytest.raises(RuntimeError, match="all children refused"):
        allocate(host_worker)
    assert captured[0].views_released
    host_worker.close()
    assert captured[0].closed and host_worker._cleanup_journal.empty


def test_direct_host_buffer_uses_existing_domain_and_rejects_live_release(monkeypatch):
    worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
    chip = MagicMock()
    chip.comm_init.return_value = 9

    def allocate_domain(_comm, _allocation, ranks, rank, size, committed):
        assert (ranks, rank, size) == ([0], 0, 64)
        ctypes.c_uint64.from_address(committed).value = 1
        return 1, 2, 64

    chip.comm_alloc_domain_windows.side_effect = allocate_domain
    chip.comm_attach_domain_host.return_value = 8192
    worker._chip_worker = chip
    monkeypatch.setattr(worker, "_device_control_admission", lambda _api: nullcontext())
    monkeypatch.setattr(worker, "_comm_plan_rootinfo_path", lambda: "/test/direct-host")
    buffer = worker._create_direct_host_buffer(128, HostWindowSpec(128))
    resource = worker._direct_host_domains[buffer.identity][0]
    native = buffer.tensor((32,), DataType.FLOAT32)
    mapped = resource.map_view(native, 0)
    assert native.mapping_ref == 0
    assert mapped.buffer.identity == native.buffer.identity
    assert worker._chip_import_registry is not None
    assert worker._chip_import_registry.resolve_tensor(mapped).base == 8192
    worker._chip_run_touched_identities[1] = {buffer.identity}
    with pytest.raises(RuntimeError, match="in-flight"):
        worker._release_direct_host_resource(buffer.identity)
    assert not resource.closed
    worker._chip_run_touched_identities.clear()
    worker._release_direct_host_resource(buffer.identity)
    chip.comm_release_domain_windows.assert_called_once()
    assert buffer.closed
    assert not worker._direct_host_domains


def test_direct_host_buffer_size_mismatch_has_no_side_effects():
    worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
    with pytest.raises(ValueError, match="requested buffer size"):
        worker._create_direct_host_buffer(128, HostWindowSpec(64))
    assert not worker._direct_host_domains
