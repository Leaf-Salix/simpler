# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import threading
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from simpler import host_consumer as hc
from simpler import worker as wm
from simpler.buffer import AccessMode, buffer_reference
from simpler.comm_domain import CommDomainRef
from simpler.global_comm_domain import GlobalDomainPhase
from simpler.orchestrator import _callback_frame_for, _callback_run
from test_host_domain_attachments import _commit, imported, planner  # noqa: F401 -- shared pytest fixtures


def _noop(_payload):
    pass


@pytest.fixture
def consumer(imported, planner, monkeypatch):  # noqa: F811 -- pytest injects the imported fixtures
    worker, state, command, native, pools = imported
    planner._remote_worker_specs[0] = replace(planner._remote_worker_specs[0], num_sub_workers=1)
    planner._endpoint_registry = None
    grants = planner._global_host_consumer_attachments(command)
    assert {(g.node_worker_id, g.sub_worker_id, rank) for g in grants for rank in g.provider_ranks} == {
        (0, 0, 0),
        (0, 0, 1),
    }
    command = replace(command, host_consumer_attachments=grants)
    state.command = replace(state.command, host_consumer_attachments=grants)
    _commit((worker, state, command, native, pools))
    worker._sub_pids = [os.getpid()]
    registrations, removals = [], []

    def register(fn):
        with worker._registry_lock:
            handle, _ = worker._install_registration_locked(wm._build_callable_registration(worker, fn))
        registrations.append(handle)
        return handle

    def unregister(handle):
        removals.append(handle)
        with worker._registry_lock:
            worker._rollback_handle_locked(handle)

    monkeypatch.setattr(worker, "register", register)
    monkeypatch.setattr(worker, "unregister", unregister)
    try:
        yield SimpleNamespace(
            worker=worker,
            state=state,
            peer=buffer_reference(command.backing_refs[1]),
            pools=pools,
            native=native,
            registrations=registrations,
            removals=removals,
        )
    finally:
        worker._sub_pids = []
        worker._live_handles.clear()
        worker._identity_registry.clear()
        worker._callable_registry.clear()


@pytest.mark.parametrize("profile", ["sim", "a3-fabric-v1", "unknown"])
def test_host_consumer_grants_reuse_node_decisions_without_replanning(imported, planner, monkeypatch, profile):  # noqa: F811 -- shared pytest fixtures
    command = imported[2]
    planner._remote_worker_specs = [replace(spec, num_sub_workers=2) for spec in planner._remote_worker_specs]
    planner._endpoint_registry = None
    matrix = planner._global_domain_attachment_matrix(
        command.members, (0, 1), command.window_size, profile=profile, host_nodes=(0, 1)
    )
    command = replace(command, profile=profile, attachments=matrix)
    monkeypatch.setattr(wm.BackendResolver, "plan", lambda *_args: pytest.fail("SUB repeated the node planner"))
    grants = planner._global_host_consumer_attachments(command)
    assert len(grants) == 4
    assert sum(len(g.provider_ranks) for g in grants) == (8 if profile != "unknown" else 4)
    registry = planner._get_endpoint_registry()
    for node in (0, 1):
        consumers = set()
        for grant in (g for g in grants if g.node_worker_id == node):
            consumer, host = registry.resolve_members(
                (
                    wm.at(f"L4/L3[{node}]/SUB[{grant.sub_worker_id}]", wm.HOST_CPU),
                    wm.at(f"L4/L3[{node}]", wm.HOST_CPU),
                )
            )
            assert grant.consumer == consumer.identity
            assert grant.provider_ranks == tuple(
                rank
                for rank, adapter in enumerate(command.attachments_for_node(node))
                if adapter.adapter_kind is not None
            )
            assert consumer.identity != host.identity and registry.same_node(consumer, host)
            consumers.add(consumer.identity)
        assert len(consumers) == 2


def _call(case, fn=_noop, **kwargs):
    options = dict(domain=CommDomainRef(b"creator1", 7, 3), staging_offset=16, staging_nbytes=32)
    options.update(kwargs)
    case.worker.call_host(fn, case.peer, **options)


@pytest.mark.parametrize("existing", [False, True])
def test_host_transfer_uses_committed_grants_and_only_releases_its_temporary_handle(consumer, monkeypatch, existing):
    c = consumer
    held = c.worker.register(_noop)
    c.registrations.clear()
    before = set(c.worker._live_handles)
    data = bytes(range(16))

    def serve(_control, transfer, **options):
        assert options["expected_pid"] == os.getpid()
        assert options["nbytes"] == 64 and options["capacity"] == 32
        assert transfer(True, 8, data) == b""
        assert transfer(False, 8, 16) == data

    monkeypatch.setattr(hc, "serve_host_call", serve)
    with _callback_run(1, c.worker):
        _call(c, held if existing else _noop)
    assert set(c.worker._live_handles) == before
    assert c.removals == c.registrations
    assert len(c.removals) == (0 if existing else 1)
    assert held not in c.removals
    assert c.pools[c.peer.identity] == bytes([0xA5] * 8) + data + bytes([0xA5] * 40)
    host = c.state.host_buffers[0]
    assert c.pools[host.identity] == bytes([0xCC] * 16) + data + bytes([0xCC] * 64)
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT, wm.CTRL_GLOBAL_DOMAIN_POOL_COPY, wm.CTRL_GLOBAL_DOMAIN_POOL_COPY]


@pytest.mark.parametrize(
    "mutation",
    [
        "no-callback",
        "submitted",
        "no-grant",
        "wrong-sub",
        "wrong-provider",
        "aborted",
        "no-context",
        "descriptor",
        "staging",
    ],
)
def test_host_call_rejects_invalid_admission_before_channel_and_registration(consumer, monkeypatch, mutation):
    c = consumer
    monkeypatch.setattr(hc, "serve_host_call", lambda *_a, **_kw: pytest.fail("channel admitted invalid grant"))
    options = {}
    if mutation in ("no-grant", "wrong-sub", "wrong-provider"):
        grants = c.state.command.host_consumer_attachments
        if mutation == "no-grant":
            grants = ()
        elif mutation == "wrong-sub":
            grants = tuple(replace(g, sub_worker_id=1) for g in grants)
        else:
            grants = tuple(replace(g, provider_ranks=(0,)) for g in grants)
        c.state.command = replace(c.state.command, host_consumer_attachments=grants)
    elif mutation == "aborted":
        c.state.phase = GlobalDomainPhase.ABORT
    elif mutation == "no-context":
        c.state.contexts.clear()
    elif mutation == "descriptor":
        c.peer = replace(c.peer, nbytes=32)
    elif mutation == "staging":
        options["staging_offset"] = 80
    with nullcontext() if mutation == "no-callback" else _callback_run(1, c.worker):
        if mutation == "submitted":
            frame = _callback_frame_for(c.worker)
            assert frame is not None
            frame.has_submitted_task = True
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            _call(c, **options)
    assert not c.registrations and not c.removals
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]


@pytest.mark.parametrize(
    "access,writing,offset,size,error",
    [
        (AccessMode.READ, True, 0, 1, "access grant"),
        (AccessMode.WRITE, False, 0, 1, "access grant"),
        (AccessMode.READWRITE, True, 0, 33, "capacity"),
        (AccessMode.READWRITE, False, 0, 33, "capacity"),
        (AccessMode.READWRITE, True, 64, 1, "range"),
        (AccessMode.READWRITE, False, 64, 1, "range"),
    ],
)
def test_host_transfer_rechecks_access_and_bounds_before_staging(
    consumer, monkeypatch, access, writing, offset, size, error
):
    c = consumer
    before = {identity: bytes(data) for identity, data in c.pools.items()}

    def serve(_control, transfer, **_options):
        transfer(writing, offset, b"x" * size if writing else size)

    monkeypatch.setattr(hc, "serve_host_call", serve)
    with _callback_run(1, c.worker), pytest.raises(ValueError, match=error):
        _call(c, access=access)
    assert c.removals == c.registrations and len(c.removals) == 1
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
    assert {identity: bytes(data) for identity, data in c.pools.items()} == before


@pytest.mark.parametrize("mutation", ["grant", "rank", "identity", "generation"])
def test_revocation_after_channel_entry_prevents_copy(consumer, monkeypatch, mutation):
    c = consumer
    before = {identity: bytes(data) for identity, data in c.pools.items()}

    def serve(_control, transfer, **_options):
        grants = c.state.command.host_consumer_attachments
        if mutation == "rank":
            grants = tuple(replace(g, provider_ranks=(0,)) for g in grants)
        elif mutation == "identity":
            grants = tuple(
                replace(g, consumer=replace(g.consumer, endpoint_id=g.consumer.endpoint_id + 1)) for g in grants
            )
        else:
            grants = ()
        c.state.command = replace(
            c.state.command,
            **({"generation": 4} if mutation == "generation" else {"host_consumer_attachments": grants}),
        )
        transfer(True, 0, b"x")

    monkeypatch.setattr(hc, "serve_host_call", serve)
    with _callback_run(1, c.worker), pytest.raises((ValueError, KeyError)):
        _call(c)
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
    assert {identity: bytes(data) for identity, data in c.pools.items()} == before
    assert c.removals == c.registrations and len(c.removals) == 1


def test_native_copy_failure_preserves_abort_and_cleanup_debt(consumer, monkeypatch):
    c = consumer

    def fail_copy(_kind, _chip, control, _payload, _timeout):
        assert control == wm.CTRL_GLOBAL_DOMAIN_POOL_COPY
        c.native.append(control)
        raise RuntimeError("native completion unknown")

    monkeypatch.setattr(c.worker._worker, "control_payload", fail_copy)
    monkeypatch.setattr(hc, "serve_host_call", lambda _control, transfer, **_options: transfer(False, 0, 1))
    with _callback_run(1, c.worker), pytest.raises(RuntimeError, match="native completion unknown"):
        _call(c)
    assert c.state.phase is GlobalDomainPhase.ABORT
    assert c.worker._ordered_cleanup_error is not None and not c.worker._cleanup_journal.empty
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT, wm.CTRL_GLOBAL_DOMAIN_POOL_COPY]


def test_unsettled_control_retains_callable_and_blocks_future_work(consumer, monkeypatch):
    c = consumer
    error = hc.HostCallUnsettledError(threading.current_thread())

    def unsettled(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(hc, "serve_host_call", unsettled)
    with _callback_run(1, c.worker), pytest.raises(hc.HostCallUnsettledError) as exc:
        _call(c)
    assert exc.value is error and len(c.registrations) == 1 and not c.removals
    assert c.registrations[0]._handle_id in c.worker._live_handles
    assert c.worker._ordered_cleanup_error is not None and not c.worker._cleanup_journal.empty
    with _callback_run(2, c.worker), pytest.raises(RuntimeError, match="cleanup failed"):
        _call(c)
    assert len(c.registrations) == 1 and c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]


@pytest.mark.parametrize("kind", ["chip", "remote"])
def test_non_python_callable_is_rejected_before_temporary_registration(consumer, monkeypatch, kind):
    c = consumer
    target = (
        wm.ChipCallable.build(signature=[], func_name="unused", binary=b"\0", children=[])
        if kind == "chip"
        else wm.RemoteCallable("builtins:len")
    )
    monkeypatch.setattr(hc, "serve_host_call", lambda *_a, **_kw: pytest.fail("invalid callable reached channel"))
    with _callback_run(1, c.worker), pytest.raises(TypeError, match="Python callable"):
        _call(c, target)
    assert not c.registrations and c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]


def test_requested_access_cannot_exceed_canonical_buffer_grant(consumer, monkeypatch):
    c = consumer
    c.peer = replace(c.peer, access=AccessMode.READ)
    c.state.command = replace(c.state.command, backing_refs=(c.state.command.backing_refs[0], c.peer.to_descriptor()))
    monkeypatch.setattr(hc, "serve_host_call", lambda *_a, **_kw: pytest.fail("write admitted for read-only buffer"))
    with _callback_run(1, c.worker), pytest.raises(ValueError, match="access exceeds"):
        _call(c, access=AccessMode.WRITE)
    assert not c.registrations and c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]


@pytest.mark.parametrize("acknowledged", [False, True])
def test_control_failure_retains_callable_only_when_completion_is_unknown(consumer, monkeypatch, acknowledged):
    c = consumer
    message = "control_generic failed on child: user exception" if acknowledged else "control transport timeout"

    def control_failure(kind, sub, operation, _payload, _timeout):
        assert kind is wm.WorkerType.SUB and sub == 0 and operation == wm._CTRL_HOST_ACCESS
        raise RuntimeError(message)

    monkeypatch.setattr(c.worker._worker, "control_payload", control_failure)
    monkeypatch.setattr(hc, "serve_host_call", lambda control, _transfer, **_kw: control("/unused", bytes(16)))
    with _callback_run(1, c.worker), pytest.raises(RuntimeError, match=message):
        _call(c)
    assert len(c.registrations) == 1
    assert c.removals == (c.registrations if acknowledged else [])
    assert (c.worker._ordered_cleanup_error is None) is acknowledged
    assert c.worker._cleanup_journal.empty is acknowledged


def test_revoking_other_rank_preserves_active_target_grant(consumer, monkeypatch):
    c = consumer

    def serve(_control, transfer, **_options):
        assert transfer(False, 0, 1) == b"\xa5"
        c.state.command = replace(
            c.state.command,
            host_consumer_attachments=tuple(
                replace(g, provider_ranks=(1,)) for g in c.state.command.host_consumer_attachments
            ),
        )
        assert transfer(True, 0, b"x") == b""
        assert transfer(False, 0, 1) == b"x"

    monkeypatch.setattr(hc, "serve_host_call", serve)
    with _callback_run(1, c.worker):
        _call(c)
    assert c.removals == c.registrations and len(c.removals) == 1
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT] + [wm.CTRL_GLOBAL_DOMAIN_POOL_COPY] * 3


def test_two_subs_keep_disjoint_rank_admission(consumer, monkeypatch):
    c = consumer
    grant = c.state.command.host_consumer_attachments[0]
    c.state.command = replace(
        c.state.command,
        host_consumer_attachments=(
            replace(grant, provider_ranks=(0,)),
            replace(grant, sub_worker_id=1, consumer=replace(grant.consumer, endpoint_id=99), provider_ranks=(1,)),
        ),
    )
    c.worker._sub_pids = [os.getpid(), os.getpid()]
    local = buffer_reference(c.state.command.backing_refs[0])
    admitted = []
    monkeypatch.setattr(hc, "serve_host_call", lambda *_args, **_kw: admitted.append(c.peer.identity))
    with _callback_run(1, c.worker):
        _call(c, sub_worker=1)
        with pytest.raises(ValueError, match="attachment"):
            _call(c, sub_worker=0)
        c.peer = local
        _call(c, sub_worker=0)
        with pytest.raises(ValueError, match="attachment"):
            _call(c, sub_worker=1)
    assert len(c.registrations) == len(c.removals) == 2
    assert admitted == [c.state.command.backing_refs[1].identity, local.identity]
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]


@pytest.mark.parametrize("phase", [GlobalDomainPhase.IMPORT, GlobalDomainPhase.COMMIT])
@pytest.mark.parametrize("mutation", ["ranks", "identity", "members", "local-backing", "descriptor"])
def test_later_phases_cannot_replace_prepared_grants_or_provider_bindings(imported, planner, phase, mutation):  # noqa: F811 -- shared pytest fixtures
    worker, state, command, native, pools = imported
    planner._remote_worker_specs[0] = replace(planner._remote_worker_specs[0], num_sub_workers=1)
    planner._endpoint_registry = None
    grants = planner._global_host_consumer_attachments(command)
    state.command = replace(state.command, host_consumer_attachments=grants)
    command = replace(command, host_consumer_attachments=grants)
    if phase is GlobalDomainPhase.COMMIT:
        worker._import_global_domain_node(command, 0)
    changed = replace(command, phase=phase)
    grant = grants[0]
    if mutation == "ranks":
        changed = replace(changed, host_consumer_attachments=(replace(grant, provider_ranks=(0,)),))
    elif mutation == "identity":
        changed = replace(
            changed, host_consumer_attachments=(replace(grant, consumer=replace(grant.consumer, endpoint_id=99)),)
        )
    elif mutation == "members":
        changed = replace(changed, members=(command.members[0], replace(command.members[1], node_worker_id=2)))
    elif mutation == "local-backing":
        wrong = wm._domain_backing_descriptor(wm.CanonicalIdentity(b"foreign1", 99, 1), 64)
        changed = replace(changed, backing_refs=(wrong, command.backing_refs[1]))
    else:
        changed = replace(
            changed, descriptors=(replace(command.descriptors[0], handle=b"foreign"), *command.descriptors[1:])
        )
    before = list(native)
    with pytest.raises((ValueError, RuntimeError)):
        if phase is GlobalDomainPhase.IMPORT:
            worker._import_global_domain_node(changed, 0)
        else:
            worker._commit_global_domain_node(changed)
    assert native == before
    assert state.command.host_consumer_attachments == grants
    assert state.phase is not GlobalDomainPhase.COMMIT


@pytest.mark.parametrize("canonical", [False, True])
def test_local_rank_copy_uses_its_materialized_descriptor_and_preserves_guards(consumer, monkeypatch, canonical):
    c = consumer
    c.peer = buffer_reference(c.state.command.backing_refs[0])
    if not canonical:
        c.peer = c.worker._get_domain(CommDomainRef(b"creator1", 7, 3)).member(0).buffers["payload"]
    before = bytes(c.pools[c.peer.identity])
    data = bytes(range(16))

    def serve(_control, transfer, **_options):
        assert transfer(True, 8, data) == b""
        assert transfer(False, 8, len(data)) == data

    monkeypatch.setattr(hc, "serve_host_call", serve)
    with _callback_run(1, c.worker):
        _call(c)
    assert c.pools[c.peer.identity] == before[:8] + data + before[24:]
    host = c.state.host_buffers[0]
    assert c.pools[host.identity] == bytes([0xCC] * 16) + data + bytes([0xCC] * 64)
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT] + [wm.CTRL_GLOBAL_DOMAIN_POOL_COPY] * 2
    assert c.removals == c.registrations


def test_local_rank_copy_rejects_changed_mapping_before_native_copy(consumer, monkeypatch):
    c = consumer
    c.peer = buffer_reference(c.state.command.backing_refs[0])
    before = bytes(c.pools[c.peer.identity])

    def serve(_control, transfer, **_options):
        context = c.state.contexts[0]
        context.buffers["payload"] = replace(context.buffers["payload"], nbytes=32)
        transfer(False, 0, 1)

    monkeypatch.setattr(hc, "serve_host_call", serve)
    with _callback_run(1, c.worker), pytest.raises(ValueError, match="granted Buffer"):
        _call(c)
    assert c.pools[c.peer.identity] == before
    assert c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
    assert c.removals == c.registrations


def test_local_mapping_cannot_expand_canonical_access(consumer, monkeypatch):
    c = consumer
    c.peer = c.worker._get_domain(CommDomainRef(b"creator1", 7, 3)).member(0).buffers["payload"]
    readonly = replace(buffer_reference(c.state.command.backing_refs[0]), access=AccessMode.READ).to_descriptor()
    c.state.command = replace(c.state.command, backing_refs=(readonly, c.state.command.backing_refs[1]))
    monkeypatch.setattr(hc, "serve_host_call", lambda *_a, **_kw: pytest.fail("write admitted through local mapping"))
    with _callback_run(1, c.worker), pytest.raises(ValueError, match="access exceeds"):
        _call(c, access=AccessMode.WRITE)
    assert not c.registrations and c.native == [wm.CTRL_GLOBAL_DOMAIN_IMPORT]
