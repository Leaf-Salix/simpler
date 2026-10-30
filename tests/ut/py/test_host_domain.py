# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import struct
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from simpler import host_domain_backend
from simpler.buffer import AddressSpace, ImportContext, ImportRegistry
from simpler.comm_endpoints import DEVICE_AICPU, AdapterProfile
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_PROFILE_A3_FABRIC,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainDescriptor,
    GlobalDomainReleaseStage,
)
from simpler.host_domain_backend import A3HostBacking, host_attachment_profile
from simpler.task_interface import (
    CommBufferSpec,
    DataType,
    HostWindowSpec,
    TaskArgs,
    TensorTransfer,
    _materialize_chip_task_args,
)
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
            assert host_worker._prepare_buffer_references(args, None, 1 - rank) is None
            with pytest.raises(ValueError, match="another node or chip"):
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
        orch.copy_to_host_domain(domain, data, offset=64)
        assert orch.copy_from_host_domain(domain, len(data), offset=64) == data
        orch.copy_to_domain(domain, 0, data, buffer="output")
        assert orch.copy_from_domain(domain, 0, len(data), buffer="output") == data
        assert orch.get_or_allocate_domain(**request()) is domain
        with pytest.raises(ValueError, match="configuration differs"):
            orch.get_or_allocate_domain(**request(host_window=HostWindowSpec(256)))
        with pytest.raises(ValueError, match="exceeds"):
            orch.copy_to_host_domain(domain, data, offset=68)
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
    state = host_worker._global_node_domains[domain.domain_id]
    export = state.descriptors[len(domain.workers)]
    assert domain.host().nbytes == 128
    assert export.mapping_size % os.sysconf("SC_PAGE_SIZE") == 0
    assert export.mapping_size >= domain.host().nbytes
    assert host_attachment_profile(export, "a2a3sim") == AdapterProfile.HOST_SHM_MAP
    with pytest.raises(ValueError, match="does not match"):
        host_attachment_profile(export, "a2a3")
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
    state = host_worker._global_node_domains[domain.domain_id]
    original = state.host_owner.release
    attempts = []

    def fail_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("injected Host owner free failure")
        original()

    monkeypatch.setattr(state.host_owner, "release", fail_once)
    with pytest.raises(RuntimeError, match="injected Host owner"):
        domain.release()
    assert state.views_released and state.owners_released
    assert host_worker._domain_free_results[domain.allocation_id] is None
    host_worker.close()
    assert len(attempts) == 2
    assert domain.freed and host_worker._cleanup_journal.empty


def test_failed_creation_without_any_import_releases_owner(host_worker, monkeypatch):
    states = []
    original = host_worker._import_global_domain_node

    def refuse(command, node, **_kwargs):
        states.append(host_worker._global_node_domains[command.domain_id])
        raise RuntimeError("injected import failure")

    monkeypatch.setattr(host_worker, "_import_global_domain_node", refuse)
    with pytest.raises(RuntimeError, match="injected import failure"):
        allocate(host_worker)
    assert states[0].views_released and states[0].host_owner is None
    assert not host_worker._global_node_domains
    monkeypatch.setattr(host_worker, "_import_global_domain_node", original)


def test_direct_host_buffer_uses_existing_domain_and_rejects_live_release(monkeypatch):
    worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
    chip = MagicMock()
    descriptor = GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, GLOBAL_DOMAIN_PROFILE_IDS["sim"], 0, 1, 64, b"test")
    chip.comm_global_domain_prepare.return_value = (descriptor.encode(), 4096, 64)
    chip.comm_global_domain_window.return_value = (8192, 128)
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
    args = TaskArgs()
    args.add_tensor(native, transfer=TensorTransfer.H2D)
    args.add_tensor(mapped, transfer=TensorTransfer.NONE)
    worker._child_prov_check_dispatch_locked(args, 0, api="test")
    resolved = _materialize_chip_task_args(args, worker._chip_import_registry)
    assert resolved.tensor(0).data == buffer.base
    assert resolved.tensor(0).address_space == AddressSpace.HOST
    assert resolved.tensor(1).data == 8192
    assert resolved.tensor(1).address_space == AddressSpace.DEVICE
    with pytest.raises(ValueError, match="target worker"):
        worker._child_prov_check_dispatch_locked(args, 1, api="test")
    worker._chip_run_touched_identities[1] = {buffer.identity}
    with pytest.raises(RuntimeError, match="in-flight"):
        worker._release_direct_host_resource(buffer.identity)
    assert not resource.closed
    worker._chip_run_touched_identities.clear()
    worker._release_direct_host_resource(buffer.identity)
    assert [call.args[1] for call in chip.comm_global_domain_release.call_args_list] == [
        GlobalDomainReleaseStage.VIEWS,
        GlobalDomainReleaseStage.OWNERS,
    ]
    chip.comm_init.assert_not_called()
    chip.comm_global_domain_prepare.assert_called_once_with(1, 0, 1, 64, GLOBAL_DOMAIN_PROFILE_IDS["sim"], True)
    assert buffer.closed
    assert not worker._direct_host_domains
    with pytest.raises(ValueError, match="not a live allocation"):
        worker._child_prov_check_dispatch_locked(args, 0, api="test")
    with pytest.raises(ValueError, match="no granted attachment"):
        _materialize_chip_task_args(args, worker._chip_import_registry)


def test_direct_host_buffer_size_mismatch_has_no_side_effects():
    worker = Worker(level=2, platform="a2a3sim", runtime="tensormap_and_ringbuffer", device_id=0)
    with pytest.raises(ValueError, match="requested buffer size"):
        worker._create_direct_host_buffer(128, HostWindowSpec(64))
    assert not worker._direct_host_domains


def test_released_mapping_cannot_retain_dispatch_permission(host_worker):
    domain = allocate(host_worker)
    source = domain.host().tensor((16,), DataType.FLOAT32)
    mapped = domain.map_view(source, domain_rank=0)
    domain.release()
    args = TaskArgs()
    args.add_tensor(mapped, transfer=TensorTransfer.NONE)
    with host_worker._child_prov_lock, pytest.raises(ValueError, match="not a live"):
        host_worker._child_prov_check_dispatch_locked(args, 1, api="test")


def test_host_mapping_rejects_foreign_and_mapped_tensors(host_worker):
    domain = allocate(host_worker)
    source = domain.host().tensor((16,), DataType.FLOAT32)
    mapped = domain.map_view(source, domain_rank=0)
    with pytest.raises(ValueError, match="native Tensor"):
        domain.map_view(mapped, domain_rank=0)
    domain.release()
    replacement = allocate(host_worker)
    with pytest.raises(ValueError, match="exact backing"):
        replacement.map_view(source, domain_rank=0)
    replacement.release()


def test_local_host_domain_id_collision_keeps_existing_domain(host_worker):
    first = allocate(host_worker)
    state = host_worker._global_node_domains[first.domain_id]
    next_id = host_worker._next_alloc_id
    host_worker._next_alloc_id -= 1
    try:
        with pytest.raises(RuntimeError, match="conflicts with a live global domain"):
            allocate(host_worker, name="collision")
        assert host_worker._global_node_domains[first.domain_id] is state
        assert first.host().nbytes == 128
    finally:
        host_worker._next_alloc_id = next_id
        first.release()


def test_worker_close_releases_retained_local_host_domain(host_worker):
    domain = allocate(host_worker)
    host_worker.close()
    assert domain.freed
    assert not host_worker._global_node_domains


@pytest.mark.parametrize("windows", [{}, {0: HostWindowSpec(128)}])
def test_local_domain_rejects_global_host_windows(host_worker, windows):
    with pytest.raises(ValueError, match="host_windows requires an L4"):
        allocate(host_worker, host_window=None, host_windows=windows)
    with pytest.raises(ValueError, match="mutually exclusive"):
        allocate(host_worker, host_windows=windows)
    assert not host_worker._global_node_domains
    assert not host_worker._live_domains
