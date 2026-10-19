# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified L3/L4 communication domains: one API, registry, handle type and run-scoped lifecycle.

The L3 backend runs against fake chips that commit every window and record each release; the L4
backend runs against fake L3 nodes. Both drive the real allocation, admission and run-fence paths.
"""

from __future__ import annotations

import struct
import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import simpler.worker as worker_mod
from simpler.buffer import ImportContext, ImportRegistry
from simpler.comm_domain import ChipDomainMapping, ChipDomainStore, domain_window_location
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_VERSION,
    CommBufferDefinition,
    GlobalDomainBuffer,
    GlobalDomainCatalog,
    GlobalDomainCommand,
    GlobalDomainDescriptor,
    GlobalDomainMember,
    GlobalDomainPhase,
    decode_comm_init,
    decode_copy_command,
    decode_domain_command,
    encode_comm_init_result,
    encode_copy_result,
    encode_descriptor_table,
    encode_domain_command,
    resolve_global_comm_capability,
)
from simpler.orchestrator import Orchestrator
from simpler.remote_l3_protocol import ControlName
from simpler.task_interface import CommBufferSpec, CommDomainHandle, CommDomainRef, DataType, TaskArgs
from simpler.worker import RunHandle, Worker

from tests.ut.py.test_global_comm_domain import _failure_injection_worker

SPEC = CommBufferSpec("scratch", "float32", 16, 64)


def _orch(worker: Worker) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch._worker = worker
    orch._o = None
    return orch


class _Runs:
    """Builds and retires runs through the Worker's own graph-build and fence-finalization paths."""

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.unproven: set[int] = set()
        self.next_id = 1
        worker._orch = cast(
            Any,
            SimpleNamespace(
                _o=None,
                _release_run=lambda _run_id: None,
                _run_domain_use_settled=lambda run_id: run_id not in self.unproven,
            ),
        )

    def build(self, fn):
        resources = worker_mod._RunResources()
        run_id = self.next_id
        self.next_id += 1
        self.worker._building_run_resources = resources
        try:
            value = fn(_orch(self.worker))
        finally:
            self.worker._building_run_resources = None
        return value, (run_id, resources)

    def finish(self, run, error: BaseException | None = None):
        run_id, resources = run
        handle = RunHandle(self.worker, run_id, (), resources)
        self.worker._accepted_run_handles.add(handle)
        return self.worker._finalize_run_handle(handle, run_id, error)


class _FakeChips:
    """Chip children that commit every requested window and record every release."""

    def __init__(self, worker: Worker, monkeypatch, *, mapped_size: int = 8192) -> None:
        self.mapped_size = mapped_size
        self.allocations: list[int] = []
        self.releases: list[int] = []
        self.catalogs: dict[tuple[int, int], tuple[bytes, tuple[int, ...]]] = {}
        self.fail: BaseException | None = None
        self.entered = threading.Event()
        self.proceed: threading.Event | None = None
        monkeypatch.setattr(worker, "_ensure_comm_base", lambda: None)
        monkeypatch.setattr(worker, "_dispatch_control_domain", self.dispatch)

    def dispatch(self, *, workers, request_shms, reply_shms, op, allocation_id):
        if op == "release":
            self.releases.append(allocation_id)
            return
        self.entered.set()
        if self.proceed is not None:
            assert self.proceed.wait(5)
        if self.fail is not None:
            raise self.fail
        self.allocations.append(allocation_id)
        header = worker_mod._DOMAIN_REQ_HEADER
        for chip in workers:
            request = request_shms[chip].buf
            _, _, _, _, count = header.unpack_from(request, 0)
            nbytes = struct.unpack_from(f"<{count}Q", request, header.size)
            ids = struct.unpack_from(f"<{count}Q", request, header.size + 8 * count)
            creator = bytes(request[header.size + 16 * count : header.size + 16 * count + 8])
            self.catalogs[(allocation_id, chip)] = (creator, tuple(ids))
            reply = reply_shms[chip].buf
            base = 0x100000 * (chip + 1)
            worker_mod._DOMAIN_REPLY_HEADER.pack_into(reply, 0, 1, 0xC000 + chip, base, self.mapped_size, count)
            offsets = [sum(nbytes[:index]) for index in range(count)]
            if count:
                struct.pack_into(
                    f"<{count}Q", reply, worker_mod._DOMAIN_REPLY_HEADER.size, *(base + o for o in offsets)
                )


@pytest.fixture
def l3(monkeypatch):
    worker = Worker(level=3, device_ids=[0, 1], num_sub_workers=0)
    worker._worker = cast(Any, SimpleNamespace())
    chips = _FakeChips(worker, monkeypatch)
    runs = _Runs(worker)
    try:
        yield worker, chips, runs
    finally:
        worker._worker = None
        worker.close()


def _domain(orch, *, reuse=True, **changes):
    values = {"name": "tp", "members": [1, 0], "window_size": 4096, "buffers": [SPEC], "retain_after_run": True}
    values.update(changes)
    allocate = orch.get_or_allocate_domain if reuse else orch.allocate_domain
    return allocate(**values)


def test_same_configuration_returns_the_same_handle_without_reallocating(l3):
    worker, chips, runs = l3
    (first, view), _ = runs.build(lambda orch: (_domain(orch), _domain(orch).member(0).buffers["scratch"]))
    second, _ = runs.build(lambda orch: _domain(orch))

    assert second is first and chips.allocations == [first.allocation_id]
    assert worker.live_domains == {"tp": first}
    assert view.to_descriptor() == second.member(0).buffers["scratch"].to_descriptor()
    assert first.requested_window_size == 4096 and first.mapping_size == 8192
    # Members are dense ranks in request order, never chip ids.
    assert [first.member(rank).local_worker_id for rank in (0, 1)] == [1, 0]
    assert first.local_context(0).device_ctx == 0xC001
    # Each chip's catalog is the creator's identity for the slice that rank holds.
    creator, ids = chips.catalogs[(first.allocation_id, 1)]
    assert creator == worker._owner_instance_id and ids == (view.identity.buffer_id,)
    assert first.local_context(0).buffers["scratch"].identity == view.identity
    assert runs.build(lambda orch: orch.get_domain("tp"))[0] is first
    assert runs.build(lambda orch: orch.get_domain(first.ref))[0] is first


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"members": [0, 1]}, "members"),
        ({"window_size": 8192}, "window_size"),
        ({"retain_after_run": False}, "retain_after_run"),
        ({"buffers": [replace(SPEC, dtype="int32")]}, r"buffers\[0\]\.dtype"),
        ({"buffers": [replace(SPEC, count=8)]}, r"buffers\[0\]\.count"),
        ({"buffers": [replace(SPEC, nbytes=32)]}, r"buffers\[0\]\.nbytes"),
        ({"buffers": [replace(SPEC, load_from_host=True)]}, r"buffers\[0\]\.load_from_host"),
        ({"buffers": [replace(SPEC, store_to_host=True)]}, r"buffers\[0\]\.store_to_host"),
    ],
)
def test_every_configuration_field_participates_in_reuse(l3, change, field):
    _, chips, runs = l3
    runs.build(lambda orch: _domain(orch))
    with pytest.raises(ValueError, match=field):
        runs.build(lambda orch: _domain(orch, **change))
    assert len(chips.allocations) == 1


def test_configuration_is_an_immutable_snapshot(l3):
    _, _, runs = l3
    spec = CommBufferSpec("scratch", "float32", 16, 64)
    buffers = [spec]
    handle, _ = runs.build(lambda orch: _domain(orch, buffers=buffers))
    spec.nbytes = 128
    buffers.clear()
    assert runs.build(lambda orch: _domain(orch))[0] is handle
    assert handle.buffers == (CommBufferDefinition("scratch", "float32", 16, 64, False, False),)


def test_lookup_and_duplicate_allocation_never_create(l3):
    _, chips, runs = l3
    with pytest.raises(KeyError):
        runs.build(lambda orch: orch.get_domain("tp"))
    runs.build(lambda orch: _domain(orch, reuse=False))
    with pytest.raises(ValueError, match="already live"):
        runs.build(lambda orch: _domain(orch, reuse=False))
    assert len(chips.allocations) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"members": [0, 0]},
        {"members": [2]},
        {"members": [(0, 0)]},
        {"members": [True]},
        {"window_size": 32},
        {"buffers": [SPEC, SPEC]},
    ],
)
def test_bad_request_is_refused_before_any_device_effect(l3, change):
    _, chips, runs = l3
    with pytest.raises(ValueError):
        runs.build(lambda orch: _domain(orch, **change))
    assert not chips.entered.is_set()


def test_members_accept_any_integer_spelling(l3):
    _, chips, runs = l3
    domain, _ = runs.build(_domain)
    same, _ = runs.build(lambda orch: _domain(orch, members=[np.int64(1), np.int32(0)]))
    assert same is domain and len(chips.allocations) == 1


def test_failed_creation_publishes_nothing_and_frees_the_name(l3):
    worker, chips, runs = l3
    chips.fail = RuntimeError("chip refused the window")
    with pytest.raises(RuntimeError, match="refused"):
        runs.build(lambda orch: _domain(orch))
    assert worker.live_domains == {}
    chips.fail = None
    handle, _ = runs.build(lambda orch: _domain(orch))
    assert worker.live_domains == {"tp": handle}


def test_concurrent_requests_share_one_creation(l3):
    worker, chips, runs = l3
    chips.proceed = threading.Event()
    resources = worker_mod._RunResources()
    worker._building_run_resources = resources
    orch = _orch(worker)
    results: list[Any] = []

    def create():
        try:
            results.append(_domain(orch))
        except BaseException as exc:  # noqa: BLE001
            results.append(exc)

    creator, waiter = threading.Thread(target=create), threading.Thread(target=create)
    creator.start()
    try:
        assert chips.entered.wait(5)
        with pytest.raises(KeyError):
            orch.get_domain("tp")
        with pytest.raises(ValueError, match="window_size"):
            _domain(orch, window_size=8192)
        waiter.start()
    finally:
        chips.proceed.set()
        creator.join(5)
        waiter.join(5)
        worker._building_run_resources = None
    assert len(results) == 2 and isinstance(results[0], CommDomainHandle) and results[0] is results[1]
    assert len(chips.allocations) == 1


def test_release_closes_lookup_and_submission_and_allows_rebuild(l3):
    _, chips, runs = l3
    (old, tensor), first_run = runs.build(
        lambda orch: (_domain(orch), _domain(orch).member(0).buffers["scratch"].tensor([16], DataType.FLOAT32))
    )
    old.release()
    old.release()
    args = TaskArgs()
    args.add_tensor(tensor)
    with pytest.raises(KeyError):
        runs.build(lambda orch: orch.get_domain("tp"))
    with pytest.raises(RuntimeError, match="released"):
        runs.build(lambda orch: orch.get_domain(old.ref))
    with pytest.raises(RuntimeError, match="released"):
        old.member(0)
    with pytest.raises(RuntimeError, match="released"):
        runs.build(lambda orch: orch._worker._admit_domain_uses((args,), (1,)))

    new, _ = runs.build(lambda orch: _domain(orch))
    assert new.ref != old.ref
    assert new.member(0).buffers["scratch"].identity != tensor.buffer.identity
    assert old.released and not old.freed and chips.releases == []
    runs.finish(first_run)
    assert old.freed and chips.releases == [old.allocation_id]
    with pytest.raises(KeyError):
        runs.build(lambda orch: orch.get_domain(old.ref))
    # The late cleanup of the old domain left its replacement alone.
    assert runs.build(lambda orch: orch.get_domain("tp"))[0] is new


def test_backend_free_waits_for_every_run_that_used_the_domain(l3):
    _, chips, runs = l3
    domain, first = runs.build(lambda orch: _domain(orch))
    _, second = runs.build(lambda orch: _domain(orch).release())
    runs.finish(second)
    assert domain.released and not domain.freed and chips.releases == []
    runs.finish(first)
    assert domain.freed and chips.releases == [domain.allocation_id]


def test_retained_domain_survives_its_run_and_an_unretained_one_does_not(l3):
    worker, chips, runs = l3
    (kept, dropped), run = runs.build(lambda orch: (_domain(orch), _domain(orch, name="once", retain_after_run=False)))
    runs.finish(run)
    assert not kept.released and dropped.freed
    assert worker.live_domains == {"tp": kept} and chips.releases == [dropped.allocation_id]
    _, later = runs.build(lambda orch: orch.get_domain("tp").release())
    runs.finish(later)
    assert kept.freed


def test_failed_run_closes_a_retained_domain_but_waits_for_its_other_users(l3):
    worker, chips, runs = l3
    domain, created = runs.build(lambda orch: _domain(orch))
    runs.finish(created)
    _, failing = runs.build(lambda orch: _domain(orch))
    _, surviving = runs.build(lambda orch: _domain(orch))
    runs.finish(failing, RuntimeError("chip task failed"))
    assert domain.released and not domain.freed and worker.live_domains == {}
    runs.finish(surviving)
    assert domain.freed and chips.releases == [domain.allocation_id]


def test_unproven_device_work_keeps_the_domain_mapped(l3):
    worker, chips, runs = l3
    domain, run = runs.build(lambda orch: _domain(orch, retain_after_run=False))
    runs.unproven.add(run[0])
    runs.finish(run, RuntimeError("endpoint lost"))
    assert domain.released and not domain.freed and chips.releases == []
    assert worker._domain_use_unproven
    with pytest.raises(RuntimeError, match="no further work is admitted"):
        worker._require_no_ordered_cleanup_failure("submit")
    # Close relinquishes it to tree teardown without a collective free.
    worker._release_all_live_domains()
    assert not domain.freed and chips.releases == []


def test_with_block_releases_on_exception_including_a_reuse_hit(l3):
    _, _, runs = l3
    domain, _ = runs.build(lambda orch: _domain(orch))

    def fail_inside(orch):
        with _domain(orch) as hit:
            assert hit is domain
            raise ValueError("inside")

    with pytest.raises(ValueError, match="inside"):
        runs.build(fail_inside)
    assert domain.released


def test_release_domain_refuses_a_handle_of_another_worker(l3, monkeypatch):
    worker, _, runs = l3
    other = Worker(level=3, device_ids=[0, 1], num_sub_workers=0)
    other._worker = cast(Any, SimpleNamespace())
    _FakeChips(other, monkeypatch)
    _Runs(other)
    try:
        foreign = _Runs(other).build(lambda orch: _domain(orch))[0]
        with pytest.raises(ValueError, match="this Worker created"):
            runs.build(lambda orch: orch.release_domain(foreign))
        assert not foreign.released
        mine, _ = runs.build(lambda orch: _domain(orch))
        runs.build(lambda orch: orch.release_domain(mine))
        assert mine.released
    finally:
        other._worker = None
        other.close()


def test_close_sweep_frees_every_owned_domain(l3):
    worker, chips, runs = l3
    domain, run = runs.build(lambda orch: _domain(orch))
    runs.finish(run)
    worker._release_all_live_domains()
    assert domain.freed and chips.releases == [domain.allocation_id] and worker.live_domains == {}


def test_slice_must_reach_the_chip_that_holds_its_rank(l3):
    worker, _, runs = l3

    def submit(orch, target):
        args = TaskArgs()
        args.add_tensor(_domain(orch).member(0).buffers["scratch"].tensor([16], DataType.FLOAT32))
        worker._admit_domain_uses((args,), (target,))

    with pytest.raises(ValueError, match="held by worker 1"):
        runs.build(lambda orch: submit(orch, 0))
    _, run = runs.build(lambda orch: submit(orch, 1))
    domain = worker.live_domains["tp"]
    assert run[1] in domain._users


def test_tensor_outside_the_canonical_catalog_is_refused(l3):
    worker, _, runs = l3
    domain, _ = runs.build(lambda orch: _domain(orch))
    slice_buffer = domain.member(0).buffers["scratch"]
    forged = replace(slice_buffer, identity=worker._burn_buffer_identity())
    args = TaskArgs()
    args.add_tensor(forged.tensor([16], DataType.FLOAT32))
    with pytest.raises(ValueError, match="canonical slice catalog"):
        runs.build(lambda orch: worker._admit_domain_uses((args,), (1,)))


def test_group_admission_is_atomic(l3):
    worker, _, runs = l3
    (live, closed), _ = runs.build(lambda orch: (_domain(orch), _domain(orch, name="closed")))
    closed.release()
    good, bad = TaskArgs(), TaskArgs()
    good.add_domain_ref(live.ref)
    bad.add_domain_ref(closed.ref)
    users_before = set(live._users)

    def submit(_orch):
        worker._admit_domain_uses((good, bad), (0, 1))

    with pytest.raises(RuntimeError, match="released"):
        runs.build(submit)
    assert live._users == users_before


def test_views_of_one_slice_keep_its_identity_and_their_offsets(l3):
    _, _, runs = l3
    domain, _ = runs.build(lambda orch: _domain(orch))
    slice_buffer = domain.member(1).buffers["scratch"]
    whole = slice_buffer.tensor([16], DataType.FLOAT32)
    tail = slice_buffer.tensor([8], DataType.FLOAT32, byte_offset=32)
    assert whole.buffer.identity == tail.buffer.identity
    assert (whole.byte_offset, tail.byte_offset) == (0, 32)
    ref, rank, offset = domain_window_location(tail.buffer)
    assert (ref, rank, offset) == (domain.ref, 1, 0)


def test_legacy_chip_view_accepts_a_use_and_refuses_after_release(l3):
    worker, _, runs = l3
    (domain, buffer), _ = runs.build(lambda orch: (_domain(orch), _domain(orch)[0].buffers["scratch"]))
    args = TaskArgs()
    args.add_tensor(buffer.tensor([16], DataType.FLOAT32))
    _, run = runs.build(lambda orch: worker._admit_domain_uses((args,), (0,)))
    assert run[1] in domain._users
    domain.release()
    with pytest.raises(RuntimeError, match="released"):
        runs.build(lambda orch: worker._admit_domain_uses((args,), (0,)))


def test_l3_copies_reach_the_member_chip_and_are_range_checked(l3, monkeypatch):
    worker, _, runs = l3
    sent = []

    def control_payload(_kind, chip, control, payload, _timeout):
        command = worker_mod.LOCAL_COPY_REQUEST.unpack_from(payload, 0)
        sent.append((chip, control, command[2], command[4], command[5]))
        reply = bytearray(len(payload))
        worker_mod.LOCAL_COPY_REPLY.pack_into(
            reply, 0, worker_mod.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, command[2], command[3], command[5]
        )
        return bytes(reply)

    worker._worker = cast(Any, SimpleNamespace(control_payload=control_payload))

    def copies(orch):
        domain = _domain(orch)
        orch.copy_to_domain(domain, 0, b"\1" * 8, buffer="scratch", offset=8)
        orch.copy_from_domain(domain, 1, 16)
        with pytest.raises(ValueError, match="exceeds buffer"):
            orch.copy_to_domain(domain, 0, b"\1" * 64, buffer="scratch", offset=8)
        with pytest.raises(ValueError, match="mapped window"):
            orch.copy_from_domain(domain, 0, 16, offset=8190)
        return domain

    domain, _ = runs.build(copies)
    assert sent == [
        (1, worker_mod.CTRL_LOCAL_DOMAIN_COPY_TO, domain.allocation_id, 8, 8),
        (0, worker_mod.CTRL_LOCAL_DOMAIN_COPY_FROM, domain.allocation_id, 0, 16),
    ]


# --------------------------------------------------------------------------------------------------
# L4: the same API over the brokered backend, and the borrowed view an L3 node builds from it
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def l4(monkeypatch):
    worker, resources, node_ids = _failure_injection_worker()
    worker._building_run_resources = None
    commands: list[GlobalDomainCommand] = []
    controls: list[tuple[ControlName, int]] = []

    def control(worker_id, control_name, payload):
        name = ControlName(control_name)
        controls.append((name, worker_id))
        if name is ControlName.COMM_INIT:
            profile = decode_comm_init(payload).profile
            return encode_comm_init_result(
                resolve_global_comm_capability(platform="a2a3sim", profile=profile, local_device_count=1)
            )
        if name is ControlName.ALLOC_DOMAIN:
            command = decode_domain_command(payload)
            commands.append(command)
            if command.phase is not GlobalDomainPhase.PREPARE_EXPORT:
                return b""
            return encode_descriptor_table(
                tuple(
                    GlobalDomainDescriptor(
                        version=GLOBAL_DOMAIN_VERSION,
                        profile_id=GLOBAL_DOMAIN_PROFILE_IDS[command.profile],
                        domain_rank=member.domain_rank,
                        rank_count=len(command.members),
                        mapping_size=8192,
                        handle=f"/node-{member.domain_rank}".encode(),
                    )
                    for member in command.members
                    if member.node_worker_id == worker_id
                )
            )
        if name is ControlName.COPY_FROM_DOMAIN:
            return encode_copy_result(b"\7" * decode_copy_command(payload, include_data=False).nbytes)
        return b""

    monkeypatch.setattr(worker, "_global_domain_control", control)
    runs = _Runs(worker)
    try:
        yield worker, node_ids, runs, commands, controls
    finally:
        worker._building_run_resources = None
        worker._live_global_domains.clear()
        worker._worker = None
        worker.close()


def _global_domain(orch, node_ids, **changes):
    values = {
        "name": "tp",
        "members": [(node_ids[1], 0), (node_ids[0], 0)],
        "window_size": 4096,
        "buffers": [SPEC],
        "retain_after_run": True,
    }
    values.update(changes)
    return orch.get_or_allocate_domain(**values)


def test_l4_domain_uses_the_same_api_and_ships_its_catalog(l4):
    worker, node_ids, runs, commands, controls = l4
    domain, run = runs.build(lambda orch: (_global_domain(orch, node_ids), _global_domain(orch, node_ids))[0])
    looked_up, lookup_run = runs.build(lambda orch: orch.get_domain(domain.ref))
    assert looked_up is domain
    as_lists, lists_run = runs.build(
        lambda orch: _global_domain(orch, node_ids, members=[[node_ids[1], 0], [node_ids[0], 0]])
    )
    assert as_lists is domain
    assert sum(command.phase is GlobalDomainPhase.PREPARE_EXPORT for command in commands) == 2
    assert isinstance(domain, CommDomainHandle) and domain.ref is not None
    assert domain.ref.owner_instance_id == worker._owner_instance_id
    assert domain.member(0).node_worker_id == node_ids[1] and domain.mapping_size == 8192
    with pytest.raises(ValueError, match="no mapping on this node"):
        domain.local_context(0)
    prepare = next(command for command in commands if command.phase is GlobalDomainPhase.PREPARE_EXPORT)
    catalog = prepare.catalog
    assert catalog is not None and catalog.owner_instance_id == worker._owner_instance_id
    slice_buffer = domain.member(1).buffers["scratch"]
    assert slice_buffer.identity.buffer_id == catalog.buffer_ids[1]
    assert all(command.catalog is None for command in commands if command.phase is not GlobalDomainPhase.PREPARE_EXPORT)
    copied, copy_run = runs.build(lambda orch: orch.copy_from_domain(domain, 1, 4, buffer="scratch"))
    assert copied == b"\7" * 4 and (ControlName.COPY_FROM_DOMAIN, node_ids[0]) in controls
    with pytest.raises(ValueError, match="configuration differs: members"):
        runs.build(lambda orch: _global_domain(orch, node_ids, members=[(node_ids[0], 0), (node_ids[1], 0)]))
    _, later = runs.build(lambda orch: orch.get_domain("tp").release())
    for finished in (run, lookup_run, lists_run, copy_run, later):
        assert not domain.freed
        runs.finish(finished)
    assert domain.freed and (ControlName.RELEASE_DOMAIN, node_ids[1]) in controls


def test_l4_slice_must_reach_the_node_that_holds_its_rank(l4):
    worker, node_ids, runs, _, _ = l4

    def submit(orch, target):
        args = TaskArgs()
        args.add_tensor(_global_domain(orch, node_ids).member(0).buffers["scratch"].tensor([16], DataType.FLOAT32))
        worker._admit_domain_uses((args,), (target,))

    with pytest.raises(ValueError, match=f"held by worker {node_ids[1]}"):
        runs.build(lambda orch: submit(orch, node_ids[0]))
    runs.build(lambda orch: submit(orch, node_ids[1]))


def _node_commands(catalog):
    members = (GlobalDomainMember(0, 0, 0, 0), GlobalDomainMember(1, 0, 1, 1))
    common = {
        "domain_id": 29,
        "generation": 1,
        "name": "tp",
        "profile": "sim",
        "window_size": 4096,
        "members": members,
        "buffers": (GlobalDomainBuffer("scratch", 64),),
    }
    descriptors = tuple(
        GlobalDomainDescriptor(
            version=GLOBAL_DOMAIN_VERSION,
            profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
            domain_rank=rank,
            rank_count=2,
            mapping_size=4096,
            handle=f"/node-{rank}".encode(),
        )
        for rank in range(2)
    )
    prepare = GlobalDomainCommand(phase=GlobalDomainPhase.PREPARE_EXPORT, catalog=catalog, **common)
    imported = GlobalDomainCommand(phase=GlobalDomainPhase.IMPORT, descriptors=descriptors, **common)
    commit = GlobalDomainCommand(phase=GlobalDomainPhase.COMMIT, descriptors=descriptors, **common)
    return prepare, imported, commit


def _borrowing_node(imports):
    """An L3 node whose chip 0 maps rank 1 of L4 domain 29, imported and committed."""
    catalog = GlobalDomainCatalog(b"creator1", (CommBufferDefinition("scratch", "float32", 16, 64),), (11, 12))
    prepare, imported, commit = _node_commands(catalog)
    # The PREPARE that carries the catalog reaches the node through the wire codec.
    prepare = decode_domain_command(encode_domain_command(prepare))

    class FakeNativeWorker:
        def control_payload(self, _kind, worker_id, control, payload, _timeout):
            if control == worker_mod.CTRL_GLOBAL_DOMAIN_IMPORT:
                imports.append((worker_id, bytes(payload)))
                return worker_mod.LOCAL_IMPORT_REPLY.pack(
                    worker_mod.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 29, 1, 0xC0, 0x1000, 4096
                )
            return b""

    worker = Worker(level=3, device_ids=(0,))
    worker._worker = cast(Any, FakeNativeWorker())
    worker._global_node_domains[29] = worker_mod._GlobalNodeDomainState(command=prepare, prepared_domain_ranks={1})
    worker._import_global_domain_node(imported, 1)
    worker._commit_global_domain_node(commit)
    return worker, CommDomainRef(b"creator1", 29, 1)


def test_borrowed_handle_needs_a_task_that_names_it_and_cannot_release():
    imports = []
    worker, ref = _borrowing_node(imports)
    try:
        with pytest.raises(RuntimeError, match="only inside a task that names it"):
            worker._get_domain(ref)
        # The node's own view of the domain is not its to release either.
        with pytest.raises(RuntimeError, match="borrowed"):
            worker._get_global_domain(29).release()
        with worker._task_domain_scope([ref]):
            borrowed = worker._get_domain(ref)
            assert borrowed.borrowed and isinstance(borrowed, CommDomainHandle)
            slice_buffer = borrowed.member(1).buffers["scratch"]
            assert slice_buffer.identity.owner_instance_id == b"creator1" and slice_buffer.identity.buffer_id == 12
            assert borrowed.local_context(1).device_ctx == 0xC0
            with pytest.raises(ValueError, match="no mapping on this node"):
                borrowed.local_context(0)
            with pytest.raises(RuntimeError, match="borrowed"):
                borrowed.release()
            with pytest.raises(RuntimeError, match="borrowed"):
                borrowed.__enter__()
        # The chip received this rank's slices for its resolver after the descriptor table.
        (_, payload) = imports[0]
        tail = worker_mod.LOCAL_IMPORT_REQUEST.size + 2 * worker_mod.GLOBAL_DOMAIN_DESCRIPTOR_BYTES
        assert payload[tail : tail + 8] == b"creator1"
        assert struct.unpack_from("<I3Q", payload, tail + 8) == (1, 0, 64, 12)
        worker._release_global_domain_node(worker_mod.GlobalDomainReleaseCommand(29, 1))
        with worker._task_domain_scope([ref]), pytest.raises(KeyError):
            worker._get_domain(ref)
    finally:
        worker._global_node_domains.clear()
        worker._worker = None
        worker.close()


def test_unproven_work_on_a_borrowed_domain_is_reported_up():
    worker, ref = _borrowing_node([])
    runs = _Runs(worker)
    try:

        def submit(_orch):
            with worker._task_domain_scope([ref]):
                args = TaskArgs()
                args.add_tensor(worker._get_domain(ref).member(1).buffers["scratch"].tensor([16], DataType.FLOAT32))
                worker._admit_domain_uses((args,), (0,))

        _, run = runs.build(submit)
        runs.unproven.add(run[0])
        runs.finish(run, RuntimeError("chip finalize failed"))
        # The parent's completion reads this flag; it must not report the borrowed use settled.
        assert worker._domain_use_unproven
        # A later run whose own use settled does not clear it.
        _, settled_run = runs.build(submit)
        runs.finish(settled_run)
        assert worker._domain_use_unproven
        with pytest.raises(RuntimeError, match="no further work is admitted"):
            worker._require_no_ordered_cleanup_failure("submit")
    finally:
        worker._global_node_domains.clear()
        worker._worker = None
        worker.close()


def test_catalog_rides_prepare_only():
    catalog = GlobalDomainCatalog(b"creator1", (CommBufferDefinition("scratch", "float32", 16, 64),), (11, 12))
    prepare, imported, _ = _node_commands(catalog)
    assert decode_domain_command(encode_domain_command(prepare)).catalog == catalog
    with pytest.raises(ValueError, match="only carried by PREPARE_EXPORT"):
        encode_domain_command(replace(imported, catalog=catalog))
    with pytest.raises(ValueError, match="one buffer id per rank slice"):
        encode_domain_command(replace(prepare, catalog=replace(catalog, buffer_ids=(11,))))
    with pytest.raises(ValueError, match="do not match"):
        encode_domain_command(
            replace(prepare, catalog=replace(catalog, buffers=(CommBufferDefinition("other", "float32", 16, 64),)))
        )


# --------------------------------------------------------------------------------------------------
# Chip side: a slice resolves only for a task granted its domain, on the rank's own chip
# --------------------------------------------------------------------------------------------------


def test_chip_resolves_only_granted_canonical_slices(l3):
    _, _, runs = l3
    domain, _ = runs.build(lambda orch: _domain(orch))
    rank0 = domain.member(0).buffers["scratch"].to_descriptor()
    rank1 = domain.member(1).buffers["scratch"].to_descriptor()
    store = ChipDomainStore()
    store.install(domain.ref, ChipDomainMapping(1, 0, 0x2000, 8192, 0xC0, {0: (rank0.identity, rank0.nbytes)}))
    imports = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, domain_resolver=store.resolve))
    with pytest.raises(RuntimeError, match="not accepted"):
        imports.materialize(rank0)
    store.task_refs = frozenset([domain.ref])
    assert imports.materialize(rank0).base == 0x2000
    with pytest.raises(ValueError, match="this chip maps rank 0"):
        imports.materialize(rank1)
    store.install(domain.ref, ChipDomainMapping(1, 0, 0x2000, 8192, 0xC0, {0: (rank1.identity, rank0.nbytes)}))
    with pytest.raises(ValueError, match="canonical slice catalog"):
        imports.materialize(rank0)
    store.install(domain.ref, ChipDomainMapping(1, 0, 0x2000, 32, 0xC0, {0: (rank0.identity, rank0.nbytes)}))
    with pytest.raises(ValueError, match="actual window mapping"):
        imports.materialize(rank0)
    store.revoke(domain.ref)
    with pytest.raises(RuntimeError, match="does not map"):
        imports.materialize(rank0)
    assert imports._by_identity == {}
