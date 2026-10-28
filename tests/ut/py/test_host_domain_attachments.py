# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from dataclasses import replace
from types import SimpleNamespace

import pytest
from simpler import comm_endpoints as ce
from simpler import worker as wm
from simpler.buffer import CanonicalIdentity, buffer_reference
from simpler.comm_domain import CommDomainRef
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainDescriptor,
    GlobalDomainHostWindow,
    GlobalDomainMember,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    GlobalDomainReleaseStage,
    decode_domain_command,
    encode_domain_command,
)
from simpler.task_interface import HostWindowSpec


@pytest.fixture
def planner():
    root = wm.Worker(level=4, num_sub_workers=0)
    for node in range(2):
        root.add_remote_worker(
            wm.RemoteWorkerSpec(
                endpoint=f"192.0.2.{node + 1}:19073",
                platform="a2a3sim",
                device_ids=(0,),
                comm_profile="sim",
                global_device_ranks=(node,),
            )
        )
    root._lifecycle = wm._Lifecycle.READY
    try:
        yield root
    finally:
        root.close()


def _members():
    return tuple(GlobalDomainMember(node, 0, node, node) for node in range(2))


def _matrix(root, *, profile="sim", host_nodes=(0, 1), members=None):
    return root._global_domain_attachment_matrix(
        _members() if members is None else members, (0, 1), 64, profile=profile, host_nodes=host_nodes
    )


def test_host_attachment_rejects_registry_changed_during_planning(planner, monkeypatch):
    original = wm.BackendResolver.plan

    def retire_registry(resolver, *args):
        result = original(resolver, *args)
        planner._invalidate_endpoint_registry()
        return result

    monkeypatch.setattr(wm.BackendResolver, "plan", retire_registry)
    with pytest.raises(ce.EndpointResolveError, match="stale endpoint registry"):
        _matrix(planner)


def test_host_consumer_grant_rejects_stale_registry(planner, monkeypatch):
    planner._remote_worker_specs[0] = replace(planner._remote_worker_specs[0], num_sub_workers=1)
    registry = planner._get_endpoint_registry()
    planner._invalidate_endpoint_registry()
    monkeypatch.setattr(planner, "_get_endpoint_registry", lambda: registry)
    command = SimpleNamespace(
        host_windows=(SimpleNamespace(node_worker_id=0),),
        attachments_for_node=lambda _node: (SimpleNamespace(adapter_kind=ce.AdapterKind.EXPLICIT_TRANSFER),),
    )
    with pytest.raises(ce.EndpointResolveError, match="stale endpoint registry"):
        planner._global_host_consumer_attachments(command)


@pytest.mark.parametrize("profile", ["sim", "a3-fabric-v1"])
def test_cross_node_host_attachment_is_scoped_and_wire_stable(planner, profile):
    ordinary = _matrix(planner, host_nodes=())
    bound = _matrix(planner, profile=profile)
    for index in (0, 3):
        assert bound[index] == ordinary[index]
        assert bound[index].adapter_profile is ce.AdapterProfile.HOST_VMM_COPY
    for index in (1, 2):
        assert ordinary[index].adapter_kind is None
        assert bound[index].adapter_kind is ce.AdapterKind.EXPLICIT_TRANSFER
        assert bound[index].adapter_profile is ce.AdapterProfile.REMOTE_COPY
    command = GlobalDomainCommand(
        GlobalDomainPhase.PREPARE_EXPORT,
        7,
        3,
        "host",
        profile,
        4096,
        _members(),
        (GlobalDomainBuffer("payload", 64),),
        attachments=bound,
    )
    assert decode_domain_command(encode_domain_command(command)).attachments == bound


@pytest.mark.parametrize("missing", ["host", "chip", "profile"])
def test_remote_attachment_requires_host_owner_participant_and_profile(planner, missing):
    if missing == "host":
        row = _matrix(planner, host_nodes=(1,))
    elif missing == "chip":
        row = _matrix(planner, members=(GlobalDomainMember(1, 0, 1, 0),))
    else:
        row = _matrix(planner, profile="unknown")
    remote = row[0 if missing == "chip" else 1]
    assert remote.adapter_kind is None and remote.adapter_profile is None


def test_payload_capability_does_not_admit_counter_or_override_denial(planner):
    registry = planner._get_endpoint_registry()
    provider = ce.at("L4/L3[1]/L2[0]", ce.DEVICE_AICORE)
    consumer = ce.at("L4/L3[0]", ce.HOST_CPU)
    resolved = registry.resolve_region_spec((provider, consumer), ce.SingleOwner(provider=provider))
    base = ce.DefaultRegionAccessService()
    service = wm._GlobalDomainRegionAccessService(base, profile="sim")
    resolver = ce.BackendResolver(registry, base, payload_access=service)
    plan = resolver.plan(resolved, ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=0))
    assert not isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.counter.attachments == ()
    attachment = next(a for a in plan.payload.attachments if a.role is ce.AttachmentRole.CONSUMER)
    assert attachment.adapter_profile is ce.AdapterProfile.REMOTE_COPY
    assert isinstance(
        resolver.plan(resolved, ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=8)), ce.UnsupportedRegionPlan
    )
    denied = ce.StaticRegionAccessService({})
    planner._region_access_service = denied
    assert _matrix(planner)[1].adapter_kind is None


@pytest.mark.parametrize("override", [False, True])
def test_resolver_routes_payload_and_counter_to_their_selected_services(planner, override):
    registry = planner._get_endpoint_registry()
    provider = ce.at("L4/L3[1]/L2[0]", ce.DEVICE_AICORE)
    consumer = ce.at("L4/L3[0]", ce.HOST_CPU)
    resolved = registry.resolve_region_spec((provider, consumer), ce.SingleOwner(provider=provider))
    calls = []

    class Service:
        def __init__(self, label):
            self.label = label

        def evaluate_buffer_access(self, query, candidate):
            calls.append((self.label, query, candidate))
            return ce.RegionAccessDecision(supported=True)

    base = Service("base")
    resolver = (
        ce.BackendResolver(registry, base, payload_access=Service("payload"))
        if override
        else ce.BackendResolver(registry, base)
    )
    plan = resolver.plan(resolved, ce.RegionLayoutSpec(payload_bytes=64, counter_bytes=8))
    assert not isinstance(plan, ce.UnsupportedRegionPlan)
    assert plan.payload.attachments and plan.counter.attachments
    assert [call[0] for call in calls] == (["payload", "base"] if override else ["base", "base"])
    assert calls[0][1:] == calls[1][1:]


@pytest.mark.parametrize("wrong", ["consumer", "backend"])
def test_scoped_payload_service_refuses_other_endpoint_relations(wrong):
    query = ce.BufferAccessQuery(
        ce.BackendKind.POSIX_SHM if wrong == "backend" else ce.BackendKind.VMM_SHAREABLE,
        ce.DEVICE_AICPU if wrong == "consumer" else ce.HOST_CPU,
        False,
    )
    service = wm._GlobalDomainRegionAccessService(ce.DefaultRegionAccessService(), profile="sim")
    assert not any(service.evaluate_buffer_access(query, c).supported for c in ce.buffer_adapter_candidates(query))


@pytest.fixture
def imported(planner, request):
    worker = wm.Worker(level=3, device_ids=getattr(request, "param", (0,)), num_sub_workers=0)
    worker._lifecycle = wm._Lifecycle.READY
    identity = worker._burn_buffer_identity()
    command = GlobalDomainCommand(
        GlobalDomainPhase.IMPORT,
        7,
        3,
        "host",
        "sim",
        4096,
        _members(),
        (GlobalDomainBuffer("payload", 64),),
        attachments=_matrix(planner),
        owner_instance_id=b"creator1",
        host_windows=tuple(GlobalDomainHostWindow(node, HostWindowSpec(0, 96), 12 + node) for node in range(2)),
        descriptors=tuple(
            GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, 1, slot, 2, size, b"backing")
            for slot, size in enumerate((4096, 4096, 96, 96))
        ),
        backing_refs=(
            wm._domain_backing_descriptor(identity, 64),
            wm._domain_backing_descriptor(CanonicalIdentity(b"peer0001", 10, 1), 64),
        ),
    )
    host = bytearray([0xCC] * 96)
    state = wm._GlobalNodeDomainState(
        command=replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backing_refs=()),
        prepared_domain_ranks={0},
        descriptors={0: command.descriptors[0], 2: command.descriptors[2]},
        local_window_bases={0: 4096},
        mapping_sizes={0: 4096},
        node_worker_id=0,
        backing_ids={0: (identity,)},
        host_owner=SimpleNamespace(buf=host),
    )
    worker._global_node_domains[7] = state
    calls, pools = [], {identity: bytearray(64), command.backing_refs[1].identity: bytearray([0xA5] * 64)}
    pools[wm._host_domain_buffers(command)[0].identity] = host

    class Chip:
        def control_payload(self, _kind, chip, control, payload, _timeout):
            assert chip == 0
            calls.append(control)
            if control == wm.CTRL_GLOBAL_DOMAIN_IMPORT:
                return wm.LOCAL_IMPORT_REPLY.pack(wm.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 123, 4096, 4096)
            assert control == wm.CTRL_GLOBAL_DOMAIN_POOL_COPY
            *_, dst_offset, src_offset, size = wm.LOCAL_POOL_COPY.unpack_from(payload)
            offset = wm.LOCAL_POOL_COPY.size
            dst = wm._unpack_identity_wire(payload[offset:])
            src = wm._unpack_identity_wire(payload[offset + wm.CALLABLE_HASH_DIGEST_BYTES :])
            pools[dst][dst_offset : dst_offset + size] = pools[src][src_offset : src_offset + size]
            return b""

    worker._worker = Chip()
    try:
        yield worker, state, command, calls, pools
    finally:
        worker._worker = None
        worker._global_node_domains.clear()
        worker._clear_child_prov()
        if worker._ordered_cleanup_error is not None:
            with pytest.raises(RuntimeError, match="unconfirmed"):
                worker.close()
        else:
            worker.close()


def _commit(fixture):
    worker, state, command, calls, pools = fixture
    worker._import_global_domain_node(command, 0)
    assert state.view is not None and not state.view.committed
    worker._commit_global_domain_node(replace(command, phase=GlobalDomainPhase.COMMIT))
    assert state.view.committed
    return worker, state, calls, pools


def test_endpoint_import_commit_and_bidirectional_copy_preserve_offsets_and_guards(imported):
    worker, state, calls, pools = _commit(imported)
    assert state.view.attachments[1].adapter_profile is ce.AdapterProfile.REMOTE_COPY
    local = state.host_buffers[0]
    peer = buffer_reference(state.command.backing_refs[1])
    part = worker._local_host_part(CommDomainRef(b"creator1", 7, 3))
    data = bytes(range(16))
    part.write(8, data)
    worker.copy_domain(peer, local, worker=0, nbytes=16, dst_offset=24, src_offset=8)
    assert pools[peer.identity] == bytes([0xA5] * 24) + data + bytes([0xA5] * 24)
    worker.copy_domain(local, peer, worker=0, nbytes=16, dst_offset=48, src_offset=24)
    result = bytearray(16)
    part.read(48, result)
    assert result == data
    assert pools[local.identity] == bytes([0xCC] * 8) + data + bytes([0xCC] * 24) + data + bytes([0xCC] * 32)
    assert calls == [wm.CTRL_GLOBAL_DOMAIN_IMPORT, wm.CTRL_GLOBAL_DOMAIN_POOL_COPY, wm.CTRL_GLOBAL_DOMAIN_POOL_COPY]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-row",
        "unbound-adapter",
        "uncommitted",
        "view-row",
        "view-generation",
        "context",
        "owner",
        "identity",
        "descriptor",
        "range",
        "released",
    ],
)
def test_copy_refuses_unmaterialized_or_changed_grants_before_native(imported, mutation):
    worker, state, calls, pools = _commit(imported)
    local, peer = state.host_buffers[0], buffer_reference(state.command.backing_refs[1])
    offset = 0
    if mutation == "missing-row":
        state.command = replace(state.command, attachments=())
        state.view.attachments = ()
    elif mutation == "unbound-adapter":
        state.command = replace(
            state.command,
            attachments=tuple(replace(a, adapter_kind=None, adapter_profile=None) for a in state.command.attachments),
        )
        state.view.attachments = state.command.attachments_for_node(0)
    elif mutation == "uncommitted":
        state.view._committed = False
    elif mutation == "view-row":
        state.view.attachments = tuple(
            replace(a, adapter_kind=None, adapter_profile=None) for a in state.view.attachments
        )
    elif mutation == "view-generation":
        state.view.generation += 1
    elif mutation == "context":
        state.contexts[0] = replace(state.contexts[0], domain_rank=1)
    elif mutation == "owner":
        state.host_owner = None
    elif mutation == "identity":
        peer = replace(peer, identity=CanonicalIdentity(b"peer0001", 99, 1))
    elif mutation == "descriptor":
        peer = replace(peer, nbytes=32)
    elif mutation == "range":
        offset = 60
    else:
        state.phase = GlobalDomainPhase.ABORT
    before = {identity: bytes(data) for identity, data in pools.items()}
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        worker.copy_domain(peer, local, worker=0, nbytes=16, dst_offset=offset)
    assert calls == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
    assert {identity: bytes(data) for identity, data in pools.items()} == before


@pytest.mark.parametrize("mutation", ["no-owner", "wrong-role", "direct-map"])
def test_import_binding_failure_never_publishes_view(imported, mutation):
    worker, state, command, calls, _ = imported
    if mutation == "no-owner":
        state.host_owner = None
    else:
        row = list(command.attachments)
        row[1] = replace(
            row[1],
            **(
                {"role": ce.AttachmentRole.PROVIDER}
                if mutation == "wrong-role"
                else {
                    "adapter_kind": ce.AdapterKind.DIRECT_MAP,
                    "adapter_profile": ce.AdapterProfile.HOST_SVM_MAP,
                }
            ),
        )
        command = replace(command, attachments=tuple(row))
        state.command = replace(state.command, attachments=command.attachments)
    with pytest.raises(RuntimeError, match="Host attachment"):
        worker._import_global_domain_node(command, 0)
    assert calls == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
    assert state.view is None and state.phase is GlobalDomainPhase.PREPARE_EXPORT
    with pytest.raises(RuntimeError, match="IMPORT completion"):
        worker._commit_global_domain_node(replace(command, phase=GlobalDomainPhase.COMMIT))


def test_native_import_failure_never_publishes_attachment(imported, monkeypatch):
    worker, state, command, calls, _ = imported

    def fail_import(_kind, _chip, control, _payload, _timeout):
        assert control == wm.CTRL_GLOBAL_DOMAIN_IMPORT
        calls.append(control)
        raise RuntimeError("native import failed")

    monkeypatch.setattr(worker._worker, "control_payload", fail_import)
    with pytest.raises(RuntimeError, match="native import failed"):
        worker._import_global_domain_node(command, 0)
    assert calls == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
    assert state.view is None and not state.contexts
    assert state.phase is GlobalDomainPhase.PREPARE_EXPORT
    with pytest.raises(RuntimeError, match="IMPORT completion"):
        worker._commit_global_domain_node(replace(command, phase=GlobalDomainPhase.COMMIT))


def test_unknown_copy_completion_revokes_attachment_and_blocks_retry(imported, monkeypatch):
    worker, state, calls, _ = _commit(imported)
    local, peer = state.host_buffers[0], buffer_reference(state.command.backing_refs[1])

    def unknown_copy(_kind, _chip, control, _payload, _timeout):
        assert control == wm.CTRL_GLOBAL_DOMAIN_POOL_COPY
        calls.append(control)
        raise RuntimeError("copy completion unknown")

    monkeypatch.setattr(worker._worker, "control_payload", unknown_copy)
    with pytest.raises(RuntimeError, match="copy completion unknown"):
        worker.copy_domain(peer, local, worker=0, nbytes=16)
    assert state.phase is GlobalDomainPhase.ABORT
    assert worker._ordered_cleanup_error is not None
    with pytest.raises(RuntimeError, match="cleanup failed"):
        worker.copy_domain(peer, local, worker=0, nbytes=16)
    assert calls == [wm.CTRL_GLOBAL_DOMAIN_IMPORT, wm.CTRL_GLOBAL_DOMAIN_POOL_COPY]


def test_planner_refusal_survives_import_and_commit_without_native_copy(imported, planner):
    worker, state, command, calls, pools = imported
    planner._region_access_service = ce.StaticRegionAccessService({})
    command = replace(command, attachments=_matrix(planner))
    assert command.attachments_for_node(0)[1].adapter_kind is None
    state.command = replace(state.command, attachments=command.attachments)
    _commit((worker, state, command, calls, pools))
    assert state.view.attachments == command.attachments_for_node(0)
    before = {identity: bytes(data) for identity, data in pools.items()}
    with pytest.raises(RuntimeError, match="no implemented adapter"):
        worker.copy_domain(buffer_reference(command.backing_refs[1]), state.host_buffers[0], worker=0, nbytes=16)
    assert calls == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
    assert {identity: bytes(data) for identity, data in pools.items()} == before


@pytest.mark.parametrize("imported", [(0, 1)], indirect=True)
def test_partial_two_chip_import_retains_owners_until_all_views_release(imported, monkeypatch):
    worker, state, command, events, _ = imported
    identity = worker._burn_buffer_identity()
    command = replace(
        command,
        members=(command.members[0], GlobalDomainMember(0, 1, 1, 1)),
        attachments=command.attachments_for_node(0),
        host_windows=command.host_windows[:1],
        descriptors=command.descriptors[:3],
        backing_refs=(command.backing_refs[0], wm._domain_backing_descriptor(identity, 64)),
    )
    state.command = replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backing_refs=())
    state.prepared_domain_ranks.add(1)
    state.descriptors[1] = command.descriptors[1]
    state.local_window_bases[1] = 8192
    state.mapping_sizes[1] = 4096
    state.backing_ids[1] = (identity,)
    owner = SimpleNamespace(close=lambda: events.append("host-close"), unlink=lambda: events.append("host-unlink"))
    state.host_owner = owner

    def control(_kind, chip, operation, payload, _timeout):
        if operation == wm.CTRL_GLOBAL_DOMAIN_IMPORT:
            events.append(("import", chip))
            if chip == 1:
                raise RuntimeError("second chip import failed")
            return wm.LOCAL_IMPORT_REPLY.pack(wm.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 123, 4096, 4096)
        assert operation == wm.CTRL_GLOBAL_DOMAIN_RELEASE
        stage = GlobalDomainReleaseStage(wm.LOCAL_RELEASE_REQUEST.unpack(payload)[-1])
        events.append((stage, chip))
        assert state.host_owner is owner
        return b""

    monkeypatch.setattr(worker._worker, "control_payload", control)
    with pytest.raises(RuntimeError, match="second chip import failed"):
        worker._import_global_domain_node(command, 0)
    assert set(state.contexts) == {0} and state.view is None
    imported_identity = state.contexts[0].buffers["payload"].identity
    assert imported_identity in worker._child_alloc
    assert state.host_owner is owner and worker._global_node_domains[7] is state
    assert state.phase is GlobalDomainPhase.PREPARE_EXPORT
    with pytest.raises(RuntimeError, match="IMPORT completion"):
        worker._commit_global_domain_node(replace(command, phase=GlobalDomainPhase.COMMIT))
    views = GlobalDomainReleaseCommand(7, 3, GlobalDomainReleaseStage.VIEWS)
    owners = replace(views, stage=GlobalDomainReleaseStage.OWNERS)
    with pytest.raises(RuntimeError, match="all local views released"):
        worker._release_global_domain_node(owners)
    assert events == [("import", 0), ("import", 1)]
    worker._release_global_domain_node(views)
    assert state.views_released and state.host_owner is owner
    assert imported_identity not in worker._child_alloc
    assert worker._global_node_domains[7] is state
    worker._release_global_domain_node(owners)
    assert state.host_owner is None and 7 not in worker._global_node_domains
    assert events == [
        ("import", 0),
        ("import", 1),
        (GlobalDomainReleaseStage.VIEWS, 0),
        (GlobalDomainReleaseStage.VIEWS, 1),
        (GlobalDomainReleaseStage.OWNERS, 0),
        (GlobalDomainReleaseStage.OWNERS, 1),
        "host-close",
        "host-unlink",
    ]
