# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Peer control identity, binding and failure-retention contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import pytest
from simpler import comm_region_peer as peer
from simpler.buffer import AccessMode, AddressSpace, BackendKind, BufferDescriptor, CanonicalIdentity
from simpler.comm_provider import RegionControlError, RegionExportDescriptor, RegionPartKind, RegionPartLocalView
from simpler.comm_region import RegionInstanceState
from simpler.orchestrator import _callback_run
from simpler.task_interface import TaskArgs
from simpler.worker import Worker, _RunResources


class _Lease:
    def __init__(self, handle):
        self.handle = handle
        self.close_count = 0
        self.failure = None

    def __int__(self):
        return self.handle

    def close(self):
        self.close_count += 1
        if self.failure is not None:
            raise self.failure


def _frame(
    *,
    operation=peer._PEER_IMPORT,
    target=1,
    resource=7,
    transaction=3,
    generation=1,
    owner=0,
    owner_nonce=b"owner123",
    counter_nonce=None,
    counter_id=int(RegionPartKind.COUNTER),
    counter_access=AccessMode.READWRITE,
):
    descriptors = [
        BufferDescriptor(
            CanonicalIdentity(nonce, buffer_id, generation),
            AddressSpace.HOST,
            access,
            BackendKind.POSIX_SHM,
            128,
            f"peer-test-{buffer_id}".encode(),
        )
        for nonce, buffer_id, access in (
            (owner_nonce, int(RegionPartKind.PAYLOAD), AccessMode.READWRITE),
            (counter_nonce or owner_nonce, counter_id, counter_access),
        )
    ]
    return bytearray(
        peer._PEER_HEADER.pack(b"session1", transaction, resource, owner, target, operation)
        + b"".join(bytes(peer._encode_buffer_descriptor_wire(descriptor)) for descriptor in descriptors)
        + bytes(peer._PEER_BASES.size)
    )


@pytest.fixture
def imports(monkeypatch):
    leases = []

    def allocate(_descriptor, **_kwargs):
        lease = _Lease(0x1000 * (len(leases) + 1))
        leases.append(lease)
        return lease

    monkeypatch.setattr(peer, "import_region_part", allocate)
    monkeypatch.setattr(peer, "_worker_host_mapped_region_mapped_base", int)
    monkeypatch.setattr(peer, "_worker_host_mapped_region_peek_cleanup_error", lambda _token: "")
    return peer.PeerRegionImports(1, "a2a3sim", "peer-test"), leases


def _binding():
    return (peer._REGION_MAGIC_VERSION, 7, 0x1000, 128, 0x2000, 128)


def test_peer_import_retry_binding_and_tombstone(imports):
    registry, leases = imports
    frame = _frame()
    registry.handle(frame)
    assert peer._PEER_BASES.unpack_from(frame, len(frame) - peer._PEER_BASES.size) == (0x1000, 0x2000)
    registry.handle(_frame())
    assert len(leases) == 2
    assert registry.validates(_binding())
    for index in range(6):
        altered: list[int] = list(_binding())
        altered[index] += 1
        assert not registry.validates(tuple(altered))
    registry.handle(_frame(operation=peer._PEER_CLOSE))
    assert not registry.validates(_binding())
    registry.close()
    registry.close()
    assert [lease.close_count for lease in leases] == [1, 1]
    assert not registry.holds_resources
    with pytest.raises(ValueError, match="closed"):
        registry.handle(_frame())


@pytest.mark.parametrize("changed", [{"target": 0}, {"target": 2}, {"transaction": 0}, {"resource": 0}])
def test_peer_refuses_invalid_target_or_identity_before_import(imports, changed):
    registry, leases = imports
    with pytest.raises(ValueError, match="target|identity"):
        registry.handle(_frame(**changed))
    assert leases == []


@pytest.mark.parametrize(
    "changed,message",
    [
        ({"generation": 2}, "generation"),
        ({"counter_nonce": b"owner456"}, "owner nonce"),
        ({"counter_id": int(RegionPartKind.PAYLOAD)}, "identities must differ"),
        ({"counter_access": AccessMode.READ}, "READWRITE"),
    ],
)
def test_peer_checks_descriptor_pair_before_either_import(imports, changed, message):
    registry, leases = imports
    with pytest.raises(RegionControlError, match=message):
        registry.handle(_frame(**changed))
    assert leases == []
    assert not registry.holds_resources


@pytest.mark.parametrize("changed", [{"resource": 8}, {"owner": 2}, {"owner_nonce": b"owner456"}])
def test_peer_duplicate_conflicting_descriptor_is_rejected(imports, changed):
    registry, leases = imports
    registry.handle(_frame())
    with pytest.raises(ValueError, match="conflict"):
        registry.handle(_frame(**changed))
    assert len(leases) == 2
    assert registry.validates(_binding())
    registry.close()


@pytest.mark.parametrize("failed_parts", [(0,), (1,), (0, 1)])
def test_peer_failed_close_attempts_both_views_and_keeps_debt(imports, failed_parts):
    registry, leases = imports
    registry.handle(_frame())
    for index in failed_parts:
        leases[index].failure = RuntimeError("unmap failed")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unmap failed"):
            registry.close()
    assert registry.holds_resources
    assert not registry.validates(_binding())
    assert [lease.close_count for lease in leases] == [1, 1]


@pytest.mark.parametrize("failure_part", [RegionPartKind.PAYLOAD, RegionPartKind.COUNTER])
def test_peer_partial_import_failure_never_publishes_binding(imports, monkeypatch, failure_part):
    registry, leases = imports
    allocate = peer.import_region_part

    def fail(descriptor, **kwargs):
        if kwargs["part"] is failure_part:
            raise RuntimeError("import failed")
        return allocate(descriptor, **kwargs)

    monkeypatch.setattr(peer, "import_region_part", fail)
    with pytest.raises(RuntimeError, match="import failed"):
        registry.handle(_frame())
    assert not registry.validates(_binding())
    with pytest.raises(ValueError, match="incomplete"):
        registry.handle(_frame())
    registry.handle(_frame(operation=peer._PEER_CLOSE))
    assert not registry.holds_resources
    assert [lease.close_count for lease in leases] == ([1] if failure_part is RegionPartKind.COUNTER else [])


def test_peer_import_native_cleanup_debt_keeps_owner_retained(imports, monkeypatch):
    registry, leases = imports
    allocate = peer.import_region_part

    def fail(descriptor, **kwargs):
        if kwargs["part"] is RegionPartKind.COUNTER:
            raise RuntimeError("import cleanup unknown")
        return allocate(descriptor, **kwargs)

    monkeypatch.setattr(peer, "import_region_part", fail)
    monkeypatch.setattr(peer, "_worker_host_mapped_region_peek_cleanup_error", lambda _token: "cleanup unknown")
    for operation in (peer._PEER_IMPORT, peer._PEER_IMPORT, peer._PEER_CLOSE):
        with pytest.raises(RuntimeError, match="cleanup unknown"):
            registry.handle(_frame(operation=operation))
    assert registry.holds_resources
    assert not registry.validates(_binding())
    assert [lease.close_count for lease in leases] == [1]


def test_peer_close_before_import_blocks_late_request(imports):
    registry, leases = imports
    registry.handle(_frame(operation=peer._PEER_CLOSE))
    with pytest.raises(ValueError, match="closed"):
        registry.handle(_frame())
    with pytest.raises(ValueError, match="conflict"):
        registry.handle(_frame(resource=8))
    assert leases == []
    assert not registry.holds_resources


@pytest.mark.parametrize("bases", [(0, 0x2000), (0x1000, 0x1000), (0x1000, 0x2001)])
def test_peer_invalid_local_pair_is_not_published_and_can_be_closed(imports, monkeypatch, bases):
    registry, leases = imports
    iterator = iter(bases)
    monkeypatch.setattr(peer, "_worker_host_mapped_region_mapped_base", lambda _handle: next(iterator))
    with pytest.raises((ValueError, RuntimeError), match="mapping|overlap|aligned"):
        registry.handle(_frame())
    assert not registry.validates(_binding())
    registry.handle(_frame(operation=peer._PEER_CLOSE))
    assert all(lease.close_count == 1 for lease in leases)
    assert not registry.holds_resources


@pytest.mark.parametrize("in_place", [False, True])
def test_peer_lease_rejects_changed_reply_identity(in_place):
    def control(_kind, _worker, _command, frame, _timeout):
        reply = frame if in_place else bytearray(frame)
        reply[0] ^= 1
        return reply

    worker = SimpleNamespace(
        _provider_import_device_id=lambda worker_id: worker_id,
        _worker=SimpleNamespace(control_payload=control),
        _py_control_timeout_s=1.0,
    )
    instance = SimpleNamespace(_allocation_identity=(b"session1", 3), provider_resource_id=7, worker_id=0)
    frame = _frame()
    descriptors = frame[peer._PEER_HEADER.size : -peer._PEER_BASES.size]
    descriptor = RegionExportDescriptor(
        peer._decode_buffer_descriptor_wire(bytes(descriptors[: peer._DESCRIPTOR_BYTES])),
        peer._decode_buffer_descriptor_wire(bytes(descriptors[peer._DESCRIPTOR_BYTES :])),
    )
    lease = peer.PeerRegionLease(worker, instance, descriptor, 1, 0)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        lease.import_views()


@pytest.mark.parametrize("fail_confirm", [False, True])
def test_peer_both_directions_enabled_before_confirmation(imports, monkeypatch, fail_confirm):
    consumer, leases = imports
    consumer.platform = "a2a3"
    owner = peer.PeerRegionImports(0, "a2a3", "owner-test")
    enabled = set()
    confirmed = set()

    def enable(device, other, wait=True):
        enabled.add(device)
        if wait:
            if other not in enabled:
                raise RuntimeError("peer access remained unconfirmed")
            if fail_confirm and device == 0:
                raise RuntimeError("owner confirmation failed")
            confirmed.add(device)

    monkeypatch.setattr(peer, "_region_enable_peer_access", enable)

    def control(_kind, worker_id, _command, frame, _timeout):
        (owner if worker_id == 0 else consumer).handle(frame)
        return frame

    worker = SimpleNamespace(
        _provider_import_device_id=lambda worker_id: worker_id,
        _worker=SimpleNamespace(control_payload=control),
        _py_control_timeout_s=1.0,
    )
    instance = SimpleNamespace(_allocation_identity=(b"session1", 3), provider_resource_id=7, worker_id=0)
    frame = _frame()
    raw = frame[peer._PEER_HEADER.size : -peer._PEER_BASES.size]
    descriptor = RegionExportDescriptor(
        peer._decode_buffer_descriptor_wire(bytes(raw[: peer._DESCRIPTOR_BYTES])),
        peer._decode_buffer_descriptor_wire(bytes(raw[peer._DESCRIPTOR_BYTES :])),
    )
    lease = peer.PeerRegionLease(worker, instance, descriptor, 1, 0)
    try:
        if fail_confirm:
            with pytest.raises(RuntimeError, match="owner confirmation failed"):
                lease.import_views()
        else:
            assert len(lease.import_views()) == 2
            assert confirmed == {0, 1}
    finally:
        lease.close()
    assert len(leases) == 2
    assert all(mapping.close_count == 1 for mapping in leases)
    assert not consumer.holds_resources


def test_region_binding_prefix_is_exact_and_requires_complete_descriptor():
    args = TaskArgs()
    args.add_scalar(17)
    assert peer.region_binding_prefix(args) is None
    args = TaskArgs()
    args.add_scalar(peer._REGION_MAGIC_VERSION)
    with pytest.raises(ValueError, match="six scalars"):
        peer.region_binding_prefix(args)
    for value in _binding()[1:]:
        args.add_scalar(value)
    args.add_scalar(99)
    assert peer.region_binding_prefix(args) == _binding()


def test_parent_binding_requires_current_live_region_and_exact_target():
    worker = Worker(level=3, device_ids=[0, 1], platform="a2a3sim", num_sub_workers=0)
    run = worker._building_run_resources = _RunResources()
    owner_views = (
        RegionPartLocalView(RegionPartKind.PAYLOAD, 0x1000, 128),
        RegionPartLocalView(RegionPartKind.COUNTER, 0x2000, 128),
    )
    peer_views = {
        RegionPartKind.PAYLOAD: RegionPartLocalView(RegionPartKind.PAYLOAD, 0x3000, 128),
        RegionPartKind.COUNTER: RegionPartLocalView(RegionPartKind.COUNTER, 0x4000, 128),
    }
    instance = SimpleNamespace(
        _state=RegionInstanceState.LIVE,
        data_plane_error=None,
        worker_id=0,
        consumer_worker_id=1,
        provider_resource_id=7,
        _provider_payload_view=owner_views[0],
        _provider_counter_view=owner_views[1],
        local_view=lambda part, *, worker_id=None: (
            owner_views[int(part) - 1] if worker_id == 0 else peer_views.get(part)
        ),
    )
    registry = worker._region_instance_registry
    registry._instances[id(instance)] = cast(Any, instance)
    registry._run_scopes[id(instance)] = run
    owner_args = TaskArgs()
    peer_args = TaskArgs()
    for scalar in _binding():
        owner_args.add_scalar(scalar)
    for scalar in (peer._REGION_MAGIC_VERSION, 7, 0x3000, 128, 0x4000, 128):
        peer_args.add_scalar(scalar)
    worker._validate_region_binding(owner_args, 0)
    worker._validate_region_binding(peer_args, 1)
    for args, target in ((owner_args, 1), (peer_args, 0), (peer_args, 2)):
        with pytest.raises(ValueError, match="expired|submitted chip"):
            worker._validate_region_binding(args, target)
    worker._building_run_resources = _RunResources()
    with pytest.raises(ValueError, match="expired|submitted chip"):
        worker._validate_region_binding(peer_args, 1)
    worker._building_run_resources = run
    instance.data_plane_error = RuntimeError("unknown completion")
    with pytest.raises(ValueError, match="expired|submitted chip"):
        worker._validate_region_binding(peer_args, 1)
    instance.data_plane_error = None
    instance._state = RegionInstanceState.CLOSED
    with pytest.raises(ValueError, match="expired|submitted chip"):
        worker._validate_region_binding(owner_args, 0)


def test_forwarded_binding_is_exact_and_revoked_after_callback():
    worker = Worker(level=3, device_ids=[0], platform="a2a3sim", num_sub_workers=0)
    args = TaskArgs()
    altered = TaskArgs()
    for index, value in enumerate(_binding()):
        args.add_scalar(value)
        altered.add_scalar(value + 64 if index == 2 else value)
    with pytest.raises(ValueError, match="expired|submitted chip"):
        worker._validate_region_binding(args, 0)
    worker._forwarded_region_binding = _binding()
    with pytest.raises(ValueError, match="expired|submitted chip"):
        worker._validate_region_binding(args, 0)
    with _callback_run(1, worker):
        worker._validate_region_binding(args, 0)
        with pytest.raises(ValueError, match="expired|submitted chip"):
            worker._validate_region_binding(altered, 0)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker._validate_region_binding, args, 0)
            with pytest.raises(ValueError, match="expired|submitted chip"):
                future.result(timeout=5)
    with pytest.raises(ValueError, match="expired|submitted chip"):
        worker._validate_region_binding(args, 0)
    worker._forwarded_region_binding = None
    with pytest.raises(ValueError, match="expired|submitted chip"):
        worker._validate_region_binding(args, 0)


def test_chip_checks_binding_against_its_own_views(imports):
    registry, _leases = imports
    args = TaskArgs()
    for scalar in _binding():
        args.add_scalar(scalar)
    store = SimpleNamespace(
        local_view=lambda _resource, part: RegionPartLocalView(part, 0x9000 + int(part) * 0x1000, 128)
    )
    with pytest.raises(ValueError, match="this chip"):
        registry.validate_args(args, store)
    registry.handle(_frame())
    registry.validate_args(args, store)
    registry.close()
    with pytest.raises(ValueError, match="this chip"):
        registry.validate_args(args, store)
