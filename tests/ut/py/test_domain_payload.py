# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import threading
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace

import pytest
import simpler.worker as module
from simpler.buffer import AccessMode, AddressSpace, BackendKind, BufferDescriptor, CanonicalIdentity, buffer_reference
from simpler.comm_region import PayloadPart, RegionPartSpan
from simpler.global_comm_domain import (
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainHostWindow,
    GlobalDomainMember,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    GlobalDomainReleaseStage,
)
from simpler.task_interface import ChipDomainContext, GlobalCommDomainView, HostWindowSpec


@pytest.fixture
def payload_domain():
    worker = module.Worker(level=3, num_sub_workers=0)
    command = GlobalDomainCommand(
        GlobalDomainPhase.COMMIT,
        7,
        3,
        "payload",
        "sim",
        4096,
        (GlobalDomainMember(2, 0, 0, 0), GlobalDomainMember(5, 0, 1, 1)),
        (GlobalDomainBuffer("payload", 128),),
        owner_instance_id=b"creator1",
        host_windows=(
            GlobalDomainHostWindow(2, HostWindowSpec(0, 256), 12),
            GlobalDomainHostWindow(5, HostWindowSpec(0, 256), 13),
        ),
        backing_refs=tuple(
            BufferDescriptor(
                CanonicalIdentity(owner, 10, 1),
                AddressSpace.DEVICE,
                AccessMode.READWRITE,
                BackendKind.REMOTE_SIDECAR,
                128,
            )
            for owner in (b"hbmnode2", b"hbmnode5")
        ),
    )
    hosts = module._host_domain_buffers(command)
    pools = {"local": hosts[2], "remote": hosts[5], "hbm": buffer_reference(command.backing_refs[1])}
    allocations = {buffer.identity: SharedMemory(create=True, size=buffer.nbytes) for buffer in pools.values()}
    for owner in allocations.values():
        assert owner.buf is not None
        owner.buf[:] = b"\xcc" * owner.size
    state = module._GlobalNodeDomainState(
        command,
        phase=GlobalDomainPhase.COMMIT,
        view=GlobalCommDomainView(
            name=command.name, members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
        ),
        contexts={0: ChipDomainContext(command.name, 0, 2, 0, 0, 4096, {})},
        host_owner=allocations[hosts[2].identity],
        node_worker_id=2,
        host_buffers=hosts,
    )
    fixture = SimpleNamespace(
        worker=worker, state=state, pools=pools, allocations=allocations, copies=[], releases=[], before_copy=None
    )

    def control(_kind, chip, operation, payload, _timeout):
        assert chip == 0
        if operation == module.CTRL_GLOBAL_DOMAIN_RELEASE:
            fixture.releases.append(bytes(payload))
            return b""
        assert operation == module.CTRL_GLOBAL_DOMAIN_POOL_COPY
        magic, version, domain, generation, dst_offset, src_offset, size = module.LOCAL_POOL_COPY.unpack_from(payload)
        assert (magic, version, domain, generation) == (module.LOCAL_DOMAIN_MAGIC, module.GLOBAL_DOMAIN_VERSION, 7, 3)
        begin = module.LOCAL_POOL_COPY.size
        stride = module.CALLABLE_HASH_DIGEST_BYTES
        dst, src = (module._unpack_identity_wire(payload[start : start + stride]) for start in (begin, begin + stride))
        fixture.copies.append((dst, src, dst_offset, src_offset, size))
        if fixture.before_copy is not None:
            fixture.before_copy()
        destination, source = allocations[dst], allocations[src]
        assert destination.buf is not None and source.buf is not None
        destination.buf[dst_offset : dst_offset + size] = source.buf[src_offset : src_offset + size]
        return b""

    worker._worker = SimpleNamespace(control_payload=control)
    worker._lifecycle = module._Lifecycle.READY
    worker._global_node_domains[7] = state
    allocation_id = worker._global_domain_provenance_id(7)
    with worker._child_prov_lock:
        for buffer in pools.values():
            worker._record_device_alloc(buffer, domain_allocation_id=allocation_id)
    try:
        yield fixture
    finally:
        worker._worker = None
        worker._global_node_domains.clear()
        worker._clear_child_prov()
        worker._ordered_cleanup_error = None
        worker._cleanup_journal = module.CleanupJournal()
        worker.close()
        for owner in allocations.values():
            owner.close()
            owner.unlink()


def _bind(domain, pool="remote", *, staging=RegionPartSpan(160, 48), worker=0):
    return domain.worker.domain_payload(domain.pools[pool], worker=worker, staging=staging)


def _contents(domain, pool):
    buffer = domain.pools[pool]
    allocation = domain.allocations[buffer.identity]
    assert bytes(allocation.buf[buffer.nbytes :]) == b"\xcc" * (allocation.size - buffer.nbytes)
    return bytes(allocation.buf[: buffer.nbytes])


@pytest.mark.parametrize("pool", ["remote", "hbm"])
def test_payload_read_write_preserve_offsets_lengths_and_guards(payload_domain, pool):
    domain = payload_domain
    part = _bind(domain, pool)
    assert isinstance(part, PayloadPart)
    assert part.span == RegionPartSpan(0, domain.pools[pool].nbytes)
    source = bytearray(b"left" + bytes(range(32)) + b"right")
    part.write(17, memoryview(source)[4:-5], 24)
    expected = bytearray(b"\xcc" * domain.pools[pool].nbytes)
    expected[17:41] = bytes(range(24))
    assert _contents(domain, pool) == expected
    result = bytearray(b"\xaa" * 40)
    part.read(21, memoryview(result)[5:35], 20)
    assert result == b"\xaa" * 5 + bytes(range(4, 24)) + b"\xaa" * 15
    local = bytearray(b"\xcc" * 256)
    local[160:184] = bytes(range(24))
    local[160:180] = bytes(range(4, 24))
    assert _contents(domain, "local") == local
    assert source == b"left" + bytes(range(32)) + b"right"
    assert domain.copies == [
        (domain.pools[pool].identity, domain.pools["local"].identity, 17, 160, 24),
        (domain.pools["local"].identity, domain.pools[pool].identity, 160, 21, 20),
    ]


def test_local_owner_payload_needs_no_chip_or_staging(payload_domain):
    domain = payload_domain
    part = domain.worker.domain_payload(domain.pools["local"])
    part.write(7, b"payload")
    result = bytearray(7)
    part.read(7, result)
    assert result == b"payload"
    assert _contents(domain, "local") == b"\xcc" * 7 + b"payload" + b"\xcc" * 242
    assert not domain.copies


@pytest.mark.parametrize(
    "worker,staging",
    [
        (None, RegionPartSpan(160, 48)),
        (0, None),
        (1, RegionPartSpan(160, 48)),
        (0, RegionPartSpan(-1, 48)),
        (0, RegionPartSpan(240, 48)),
        (0, RegionPartSpan(160, 0)),
    ],
)
def test_payload_binding_rejects_unavailable_route_without_side_effects(payload_domain, worker, staging):
    domain = payload_domain
    before = _contents(domain, "local")
    with pytest.raises((ValueError, RuntimeError)):
        _bind(domain, worker=worker, staging=staging)
    assert _contents(domain, "local") == before
    assert not domain.copies


@pytest.mark.parametrize("writing", [False, True])
@pytest.mark.parametrize("offset,nbytes", [(0, 49), (250, 12), (-1, 4), (0, 65)])
def test_payload_preflight_never_modifies_staging(payload_domain, writing, offset, nbytes):
    domain = payload_domain
    part = _bind(domain)
    data = bytearray(b"\xbb" * 64)
    before = _contents(domain, "local")
    with pytest.raises(ValueError):
        (part.write if writing else part.read)(offset, data, nbytes)
    assert _contents(domain, "local") == before
    assert data == b"\xbb" * 64
    assert not domain.copies


@pytest.mark.parametrize("mutation", ["extent", "identity", "access"])
def test_payload_rejects_forged_descriptor(payload_domain, mutation):
    domain = payload_domain
    target = domain.pools["remote"]
    changes = {
        "extent": {"nbytes": target.nbytes - 1},
        "identity": {
            "identity": CanonicalIdentity(
                target.identity.owner_instance_id, target.identity.buffer_id, target.identity.generation + 1
            )
        },
        "access": {"access": AccessMode.READ},
    }
    with pytest.raises((ValueError, KeyError)):
        domain.worker.domain_payload(replace(target, **changes[mutation]), worker=0, staging=RegionPartSpan(160, 48))
    assert not domain.copies


@pytest.mark.parametrize("mutation", ["generation", "owner", "aborted", "missing", "descriptor"])
def test_bound_payload_revalidates_live_domain_and_descriptor(payload_domain, mutation):
    domain = payload_domain
    part = _bind(domain, "hbm")
    command = domain.state.command
    if mutation == "generation":
        domain.state.command = replace(command, generation=command.generation + 1)
    elif mutation == "owner":
        domain.state.command = replace(command, owner_instance_id=b"creator2")
    elif mutation == "aborted":
        domain.state.phase = GlobalDomainPhase.ABORT
    elif mutation == "missing":
        domain.worker._global_node_domains.clear()
    else:
        domain.state.command = replace(
            command,
            backing_refs=(
                command.backing_refs[0],
                replace(buffer_reference(command.backing_refs[1]), nbytes=64).to_descriptor(),
            ),
        )
    before = _contents(domain, "local")
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        part.write(0, b"changed")
    assert _contents(domain, "local") == before
    assert not domain.copies


@pytest.mark.parametrize("writing", [False, True])
def test_payload_copy_failure_poisoning_has_no_false_success(payload_domain, writing):
    domain = payload_domain
    part = _bind(domain)
    result = bytearray(b"unchanged")

    def fail():
        raise RuntimeError("completion unknown")

    domain.before_copy = fail
    with pytest.raises(RuntimeError, match="completion unknown"):
        (part.write if writing else part.read)(0, result)
    assert result == b"unchanged"
    assert domain.state.phase is GlobalDomainPhase.ABORT
    assert domain.worker._ordered_cleanup_error is not None
    assert _contents(domain, "remote") == b"\xcc" * 256
    with pytest.raises((KeyError, RuntimeError)):
        part.read(0, result)
    assert len(domain.copies) == 1


def _thread(errors, action, *, name):
    def run():
        try:
            action()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    return threading.Thread(target=run, name=name, daemon=True)


@pytest.mark.parametrize("boundary", ["copy", "readback"])
def test_domain_release_waits_until_composite_payload_read_finishes(payload_domain, monkeypatch, boundary):
    domain = payload_domain
    part = _bind(domain)
    domain.allocations[domain.pools["remote"].identity].buf[:8] = b"complete"
    entered, resume, releasing = threading.Event(), threading.Event(), threading.Event()
    errors, result = [], bytearray(8)

    def pause():
        entered.set()
        assert resume.wait(3)

    if boundary == "copy":
        domain.before_copy = pause
    else:
        original_host_bytes = domain.worker._host_domain_bytes_locked

        def pause_readback(state, offset, pointer, nbytes, *, writing):
            assert not writing
            assert _contents(domain, "local")[160:168] == b"complete"
            assert result == bytes(8)
            pause()
            original_host_bytes(state, offset, pointer, nbytes, writing=writing)

        monkeypatch.setattr(domain.worker, "_host_domain_bytes_locked", pause_readback)
    original_drop = domain.worker._drop_domain_allocs

    def observe_drop(allocation):
        releasing.set()
        original_drop(allocation)

    monkeypatch.setattr(domain.worker, "_drop_domain_allocs", observe_drop)
    domain.state.prepared_domain_ranks = {0}
    reader = _thread(errors, lambda: part.read(0, result), name="payload-read")
    releaser = _thread(
        errors,
        lambda: domain.worker._release_global_domain_node(
            GlobalDomainReleaseCommand(7, 3, GlobalDomainReleaseStage.VIEWS)
        ),
        name="payload-release",
    )
    try:
        reader.start()
        assert entered.wait(3)
        assert domain.worker._child_prov_lock.locked()
        releaser.start()
        assert releasing.wait(3)
        assert not domain.releases
    finally:
        resume.set()
        for thread in (reader, releaser):
            if thread.ident is not None:
                thread.join(3)
    assert all(not thread.is_alive() for thread in (reader, releaser))
    assert not errors
    assert result == b"complete"
    assert domain.releases and domain.state.views_released
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        part.read(0, result)


def test_same_payload_serializes_staging_across_concurrent_writes(payload_domain, monkeypatch):
    domain = payload_domain
    part = _bind(domain)
    entered, resume, second_attempt = threading.Event(), threading.Event(), threading.Event()
    errors = []
    original_lock = domain.worker._child_prov_worker_lock

    def observe_lock(chip):
        if threading.current_thread().name == "payload-second":
            second_attempt.set()
        return original_lock(chip)

    def pause():
        if threading.current_thread().name == "payload-first":
            entered.set()
            assert resume.wait(3)

    monkeypatch.setattr(domain.worker, "_child_prov_worker_lock", observe_lock)
    domain.before_copy = pause
    first = _thread(errors, lambda: part.write(8, b"first"), name="payload-first")
    second = _thread(errors, lambda: part.write(32, b"second"), name="payload-second")
    try:
        first.start()
        assert entered.wait(3)
        second.start()
        assert second_attempt.wait(3)
        assert _contents(domain, "local")[160:165] == b"first"
        assert len(domain.copies) == 1
    finally:
        resume.set()
        for thread in (first, second):
            if thread.ident is not None:
                thread.join(3)
    assert all(not thread.is_alive() for thread in (first, second))
    assert not errors
    expected = bytearray(b"\xcc" * 256)
    expected[8:13], expected[32:38] = b"first", b"second"
    assert _contents(domain, "remote") == expected


@pytest.mark.parametrize("access,writing", [(AccessMode.READ, True), (AccessMode.WRITE, False)])
def test_payload_checks_granted_direction_before_touching_staging(payload_domain, access, writing):
    domain = payload_domain
    command = domain.state.command
    restricted = replace(buffer_reference(command.backing_refs[1]), access=access).to_descriptor()
    domain.state.command = replace(command, backing_refs=(command.backing_refs[0], restricted))
    domain.pools["hbm"] = buffer_reference(restricted)
    part = _bind(domain, "hbm")
    local_before = _contents(domain, "local")
    with pytest.raises(ValueError, match="access"):
        (part.write if writing else part.read)(0, bytearray(16))
    assert _contents(domain, "local") == local_before
    assert not domain.copies
    (part.read if writing else part.write)(0, bytearray(16))
    assert len(domain.copies) == 1


def test_payload_binding_captures_descriptor_not_mutable_buffer(payload_domain):
    domain = payload_domain
    target = replace(domain.pools["remote"])
    part = domain.worker.domain_payload(target, worker=0, staging=RegionPartSpan(160, 48))
    target.nbytes = 1
    part.write(8, b"captured")
    assert _contents(domain, "remote")[8:16] == b"captured"


@pytest.mark.parametrize("pool", ["local", "remote"])
def test_payload_access_rejects_after_callback_task_submission(payload_domain, pool):
    from simpler.orchestrator import _admit_task_submission, _callback_run  # noqa: PLC0415

    domain = payload_domain
    part = domain.worker.domain_payload(domain.pools["local"]) if pool == "local" else _bind(domain)
    with _callback_run(1, domain.worker):
        _admit_task_submission(domain.worker)
        with pytest.raises(RuntimeError, match="task submission"):
            part.write(0, b"reject")
    assert _contents(domain, "local") == b"\xcc" * 256
    assert not domain.copies


def test_payload_access_rejects_while_an_unrelated_run_is_live(payload_domain):
    domain = payload_domain
    part = _bind(domain)

    class PendingRun:
        _cleanup_published = False

    domain.worker._accepted_run_handles.add(PendingRun())
    try:
        with pytest.raises(RuntimeError, match="still in flight"):
            part.write(0, b"reject")
    finally:
        domain.worker._accepted_run_handles.clear()
    assert _contents(domain, "local") == b"\xcc" * 256
    assert not domain.copies


@pytest.mark.parametrize("worker,staging", [(0, None), (None, RegionPartSpan(160, 48))])
def test_local_payload_rejects_ambiguous_copy_arguments(payload_domain, worker, staging):
    domain = payload_domain
    with pytest.raises(ValueError):
        _bind(domain, "local", worker=worker, staging=staging)
    assert not domain.copies


@pytest.mark.parametrize("pool", ["local", "remote"])
def test_bound_payload_cannot_outlive_host_owner(payload_domain, pool):
    domain = payload_domain
    part = domain.worker.domain_payload(domain.pools["local"]) if pool == "local" else _bind(domain)
    domain.state.host_owner = None
    with pytest.raises(RuntimeError, match="Host DRAM owner"):
        part.write(0, b"reject")
    assert _contents(domain, "local") == b"\xcc" * 256
    assert not domain.copies


def test_bound_payload_rejects_closed_worker(payload_domain):
    domain = payload_domain
    part = _bind(domain)
    domain.worker._lifecycle = module._Lifecycle.CLOSED
    try:
        with pytest.raises(RuntimeError, match="READY"):
            part.read(0, bytearray(8))
    finally:
        domain.worker._lifecycle = module._Lifecycle.READY
    assert not domain.copies
