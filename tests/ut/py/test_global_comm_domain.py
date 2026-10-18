# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from simpler.buffer import AddressSpace, CanonicalIdentity
from simpler.comm_domain import (
    CommBufferSpec,
    CommDomainRef,
    _DomainConfig,
    _DomainMember,
    _DomainRecord,
    _DomainRegistry,
    decode_domain_metadata,
    encode_domain_metadata,
)
from simpler.comm_endpoints import (
    _ADAPTER_KIND_IDS,
    _ADAPTER_PROFILE_IDS,
    AdapterKind,
    AdapterProfile,
    AttachmentRole,
    _adapter_kind_id,
    _adapter_profile_id,
)
from simpler.global_comm_domain import (
    _ADAPTER_KIND_IDS as _DOMAIN_ADAPTER_KIND_IDS,
)
from simpler.global_comm_domain import (
    _ADAPTER_PROFILE_IDS as _DOMAIN_ADAPTER_PROFILE_IDS,
)
from simpler.global_comm_domain import (
    CTRL_GLOBAL_DOMAIN_COPY_FROM,
    CTRL_GLOBAL_DOMAIN_COPY_TO,
    CTRL_GLOBAL_DOMAIN_IMPORT,
    CTRL_GLOBAL_DOMAIN_PREPARE,
    CTRL_GLOBAL_DOMAIN_RELEASE,
    GLOBAL_DOMAIN_DESCRIPTOR_BYTES,
    GLOBAL_DOMAIN_MAX_ATTACHMENTS,
    GLOBAL_DOMAIN_MAX_BUFFERS,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_VERSION,
    GlobalCommInitCommand,
    GlobalDomainAttachment,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainCopyCommand,
    GlobalDomainDescriptor,
    GlobalDomainMember,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    decode_comm_init,
    decode_copy_command,
    decode_descriptor_table,
    decode_domain_command,
    decode_release_command,
    encode_comm_init,
    encode_comm_init_result,
    encode_copy_command,
    encode_descriptor_table,
    encode_domain_command,
    encode_release_command,
    resolve_global_comm_capability,
    validate_descriptor_table,
)
from simpler.global_comm_domain import (
    _adapter_kind_id as _domain_adapter_kind_id,
)
from simpler.global_comm_domain import (
    _adapter_profile_id as _domain_adapter_profile_id,
)

from tests.ut.py.test_worker.test_comm_endpoints import (
    FROZEN_ADAPTER_KIND_LE_U32,
    FROZEN_ADAPTER_KIND_U32,
    FROZEN_ADAPTER_PROFILE_LE_U32,
    FROZEN_ADAPTER_PROFILE_U32,
)


def _domain_command(**kwargs):
    command = GlobalDomainCommand(**kwargs)
    if command.domain_metadata:
        return command
    try:
        config = _DomainConfig.capture(
            members=[
                _DomainMember(
                    m.node_worker_id,
                    m.local_worker_id,
                    m.global_device_rank,
                    ("fixture", m.node_worker_id, m.local_worker_id),
                )
                for m in command.members
            ],
            window_size=command.window_size,
            buffers=[CommBufferSpec(b.name, "uint8", b.nbytes, b.nbytes) for b in command.buffers],
            retain_after_run=False,
            topology=(),
        )
        ref = CommDomainRef(b"testnode", command.domain_id, command.generation)
        identities = iter(range(command.domain_id * 4096 + 1, command.domain_id * 4096 + 4097))
        registry = _DomainRegistry(
            owner_instance_id=b"testnode",
            mint_identity=lambda: CanonicalIdentity(b"testnode", next(identities), command.generation),
            allocate=lambda _: None,
            destroy=lambda _: None,
        )
        record = _DomainRecord(command.name, ref, config)
        registry._catalog(record)
        return replace(command, domain_metadata=encode_domain_metadata(record))
    except (ValueError, OverflowError):
        # Invalid wire inputs must reach the encoder's own validation in their test.
        return command


def _release_command(domain_id, generation, owner_instance_id=b"testnode"):
    return GlobalDomainReleaseCommand(domain_id, generation, owner_instance_id)


def _copy_command(*args, **kwargs):
    kwargs.setdefault("owner_instance_id", b"testnode")
    return GlobalDomainCopyCommand(*args, **kwargs)


def _install_prepared_node(worker, command, *, prepared=True):
    from simpler.worker import _GlobalNodeDomainState  # noqa: PLC0415

    record = worker._domain_registry.import_domain(command.domain_metadata)
    state = _GlobalNodeDomainState(command=command, allocation_id=command.domain_id, view=record.handle)
    if prepared:
        state.prepared_domain_ranks.update(m.domain_rank for m in command.members)
    worker._global_node_domains[command.key] = state
    return state


def _members() -> tuple[GlobalDomainMember, ...]:
    return (
        GlobalDomainMember(0, 0, 3, 0),
        GlobalDomainMember(1, 0, 7, 1),
    )


def _attachments() -> tuple[GlobalDomainAttachment, ...]:
    return (
        GlobalDomainAttachment(
            node_worker_id=0,
            address_space=AddressSpace.HOST,
            role=AttachmentRole.CONSUMER,
            adapter_kind=AdapterKind.OWNER_DELEGATED_COPY,
            adapter_profile=AdapterProfile.HOST_VMM_COPY,
        ),
        GlobalDomainAttachment(
            node_worker_id=0,
            address_space=AddressSpace.HOST,
            role=AttachmentRole.CONSUMER,
            adapter_kind=AdapterKind.OWNER_DELEGATED_COPY,
            adapter_profile=AdapterProfile.HOST_VMM_COPY,
        ),
        GlobalDomainAttachment(
            node_worker_id=1,
            address_space=AddressSpace.HOST,
            role=AttachmentRole.CONSUMER,
            adapter_kind=None,
            adapter_profile=None,
        ),
        GlobalDomainAttachment(
            node_worker_id=1,
            address_space=AddressSpace.HOST,
            role=AttachmentRole.CONSUMER,
            adapter_kind=None,
            adapter_profile=None,
        ),
    )


def _descriptors() -> tuple[GlobalDomainDescriptor, ...]:
    return tuple(
        GlobalDomainDescriptor(
            version=GLOBAL_DOMAIN_VERSION,
            profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
            domain_rank=rank,
            rank_count=2,
            mapping_size=4096,
            handle=f"/simpler-test-{rank}".encode(),
        )
        for rank in range(2)
    )


def test_global_domain_version_matches_the_native_header():
    # The version is spelled twice -- GLOBAL_DOMAIN_VERSION here and COMM_GLOBAL_DOMAIN_VERSION in
    # the platform header -- and every decoder compares it for strict equality with no negotiation.
    # Bumping one alone is rejected by comm_hccl.cpp / comm_sim.cpp as a descriptor-version
    # mismatch, which names the descriptor rather than the edit that caused it. This pins the
    # pairing at the edit.
    header = Path(__file__).resolve().parents[3] / "src" / "common" / "platform_comm" / "comm.h"
    if not header.is_file():
        pytest.skip("platform_comm sources are not present in this installation")
    match = re.search(
        r"^#define\s+COMM_GLOBAL_DOMAIN_VERSION\s+(\d+)U?\s*$",
        header.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, f"COMM_GLOBAL_DOMAIN_VERSION not found in {header}"
    assert int(match.group(1)) == GLOBAL_DOMAIN_VERSION


@pytest.mark.parametrize(
    ("platform", "profile"),
    (
        ("a2a3sim", "sim"),
        ("a5sim", "sim"),
        ("a2a3", "a3-fabric-v1"),
    ),
)
def test_global_comm_capability_reports_only_implemented_backends(platform, profile):
    result = resolve_global_comm_capability(platform=platform, profile=profile, local_device_count=2)

    assert result.profile == profile
    assert result.max_ranks == 64
    assert result.descriptor_bytes == GLOBAL_DOMAIN_DESCRIPTOR_BYTES
    assert result.local_device_count == 2


@pytest.mark.parametrize(
    ("platform", "profile"),
    (
        ("a2a3", "sim"),
        ("a2a3sim", "a3-fabric-v1"),
        ("a5", "sim"),
        ("a5", "a3-fabric-v1"),
    ),
)
def test_global_comm_capability_rejects_unimplemented_backends(platform, profile):
    with pytest.raises(ValueError, match="Global CommDomain is not supported"):
        resolve_global_comm_capability(platform=platform, profile=profile, local_device_count=2)


def test_local_l3_comm_init_rejects_unsupported_capability_without_caching_topology():
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.worker import Worker, _GlobalNodeRuntime, _run_local_global_domain_control  # noqa: PLC0415

    inner_worker = Worker(level=3, num_sub_workers=0)
    runtime = _GlobalNodeRuntime(
        worker_id=0,
        device_ids=(0,),
        platform="a5",
        comm_profile="sim",
        global_device_ranks=(0,),
        node_rank=0,
        node_count=1,
        cluster_id="cluster",
        is_remote=False,
    )
    comm_inits = {}
    command = GlobalCommInitCommand(
        cluster_id="cluster",
        topology_hash="topology",
        profile="sim",
        node_rank=0,
        node_count=1,
        members=(GlobalDomainMember(0, 0, 0, 0),),
    )

    try:
        with pytest.raises(ValueError, match="Global CommDomain is not supported"):
            _run_local_global_domain_control(
                inner_worker,
                runtime,
                comm_inits,
                ControlName.COMM_INIT,
                encode_comm_init(command),
            )

        assert comm_inits == {}
    finally:
        inner_worker.close()


def test_global_domain_wire_round_trips_topology_and_descriptor_table():
    init = GlobalCommInitCommand("cluster", "topology", "sim", 0, 2, _members())
    prepare = _domain_command(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=11,
        generation=1,
        name="tp",
        profile="sim",
        window_size=2048,
        members=_members(),
        buffers=(GlobalDomainBuffer("payload", 128),),
        attachments=_attachments(),
    )
    command = _domain_command(
        phase=GlobalDomainPhase.IMPORT,
        domain_id=11,
        generation=1,
        name="tp",
        profile="sim",
        window_size=2048,
        members=_members(),
        buffers=(GlobalDomainBuffer("payload", 128),),
        descriptors=_descriptors(),
    )

    assert decode_comm_init(encode_comm_init(init)) == init
    assert decode_domain_command(encode_domain_command(prepare)) == prepare
    assert decode_domain_command(encode_domain_command(command)) == command
    assert prepare.attachments_for_node(0) == _attachments()[:2]
    assert prepare.attachments_for_node(1) == _attachments()[2:]
    assert decode_descriptor_table(encode_descriptor_table(_descriptors())) == _descriptors()
    assert GLOBAL_DOMAIN_DESCRIPTOR_BYTES == 288


def test_global_domain_wire_rejects_attachments_after_prepare():
    command = _domain_command(
        phase=GlobalDomainPhase.IMPORT,
        domain_id=11,
        generation=1,
        name="tp",
        profile="sim",
        window_size=2048,
        members=_members(),
        buffers=(),
        descriptors=_descriptors(),
        attachments=_attachments(),
    )

    with pytest.raises(ValueError, match="only carried by PREPARE_EXPORT"):
        encode_domain_command(command)


def test_global_domain_attachment_table_requires_complete_unique_receiver_rows():
    def make_command(attachments: tuple[GlobalDomainAttachment, ...]) -> GlobalDomainCommand:
        return _domain_command(
            phase=GlobalDomainPhase.PREPARE_EXPORT,
            domain_id=11,
            generation=1,
            name="attachment-shape",
            profile="sim",
            window_size=2048,
            members=_members(),
            buffers=(),
            attachments=attachments,
        )

    incomplete = make_command(_attachments()[:-1])
    with pytest.raises(ValueError, match="complete rank rows"):
        encode_domain_command(incomplete)

    duplicate_row = make_command(_attachments()[:2] * 2)
    with pytest.raises(ValueError, match="duplicate node row"):
        encode_domain_command(duplicate_row)

    assert GLOBAL_DOMAIN_MAX_ATTACHMENTS == 64 * 64


def test_global_domain_attachment_names_every_unknown_enum_field():
    # Each of the four enum-typed fields reports which field was wrong. `address_space` and `role`
    # were already wrapped; `adapter_kind` and `adapter_profile` used to surface the raw enum
    # ValueError, which names the value but not the field it came from.
    def make_command(attachment: GlobalDomainAttachment) -> GlobalDomainCommand:
        row = (attachment, replace(attachment, adapter_kind=None, adapter_profile=None))
        return _domain_command(
            phase=GlobalDomainPhase.PREPARE_EXPORT,
            domain_id=12,
            generation=1,
            name="attachment-enums",
            profile="sim",
            window_size=2048,
            members=_members(),
            buffers=(),
            attachments=row,
        )

    good = _attachments()[0]
    for field, bad_value in (
        ("address_space", 9),
        ("role", "NOT_A_ROLE"),
        ("adapter_kind", "NOT_A_KIND"),
        ("adapter_profile", "NOT_A_PROFILE"),
    ):
        with pytest.raises(ValueError, match=f"attachment {field} is unknown"):
            encode_domain_command(make_command(replace(good, **{field: bad_value})))

    # A None pair stays legal; only a half-set pair is rejected, and by its own message.
    with pytest.raises(ValueError, match="must be paired"):
        encode_domain_command(make_command(replace(good, adapter_profile=None)))


def test_adapter_numeric_authority_is_shared_with_comm_endpoints():
    assert _DOMAIN_ADAPTER_KIND_IDS is _ADAPTER_KIND_IDS
    assert _DOMAIN_ADAPTER_PROFILE_IDS is _ADAPTER_PROFILE_IDS
    live_kinds = {None: 0, **{kind.name: value for kind, value in _ADAPTER_KIND_IDS.items()}}
    live_profiles = {None: 0, **{profile.name: value for profile, value in _ADAPTER_PROFILE_IDS.items()}}
    assert live_kinds == FROZEN_ADAPTER_KIND_U32
    assert live_profiles == FROZEN_ADAPTER_PROFILE_U32
    for kind in AdapterKind:
        value = FROZEN_ADAPTER_KIND_U32[kind.name]
        assert _adapter_kind_id(kind) == value
        assert _domain_adapter_kind_id(kind) == value
        assert value.to_bytes(4, "little") == FROZEN_ADAPTER_KIND_LE_U32[kind.name]
    for profile in AdapterProfile:
        numbered = _ADAPTER_PROFILE_IDS.get(profile)
        if numbered is None:
            assert profile.name not in FROZEN_ADAPTER_PROFILE_U32
            with pytest.raises(ValueError, match="adapter_profile is unknown"):
                _adapter_profile_id(profile)
            with pytest.raises(ValueError, match="global domain attachment adapter_profile is unknown"):
                _domain_adapter_profile_id(profile)
            continue
        value = FROZEN_ADAPTER_PROFILE_U32[profile.name]
        assert numbered == value
        assert _adapter_profile_id(profile) == value
        assert _domain_adapter_profile_id(profile) == value
        assert value.to_bytes(4, "little") == FROZEN_ADAPTER_PROFILE_LE_U32[profile.name]


def test_release_and_copy_commands_round_trip_and_reject_a_missized_release():
    """RELEASE is the one command whose decoder has no length-delimited field to disagree on, so
    its fixed size is the whole frame check: a blob of any other length is rejected outright.
    """
    release = _release_command(11, 1)
    assert decode_release_command(encode_release_command(release)) == release

    copy = _copy_command(11, 1, 0, 0, 4, b"abcd")
    assert decode_copy_command(encode_copy_command(copy, include_data=True), include_data=True) == copy

    with pytest.raises(ValueError, match="release size mismatch"):
        decode_release_command(encode_release_command(release) + b"\0" * 4)


def test_global_domain_encode_rejects_too_many_buffers():
    command = _domain_command(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=11,
        generation=1,
        name="tp",
        profile="sim",
        window_size=GLOBAL_DOMAIN_MAX_BUFFERS + 1,
        members=_members(),
        buffers=tuple(GlobalDomainBuffer(f"payload-{index}", 1) for index in range(GLOBAL_DOMAIN_MAX_BUFFERS + 1)),
    )

    with pytest.raises(ValueError, match="buffer count exceeds maximum"):
        encode_domain_command(command)


def _failure_injection_worker(*, platform: str = "a2a3sim", profile: str = "sim", hosts=None):
    """Two remote L3 nodes under one L4.

    ``hosts`` defaults to both nodes on one host, which is what the failure-injection tests want
    (they exercise phase rollback, not topology). Endpoint capability is decided per node identity,
    so a test that cares whether two nodes are on the *same* machine passes distinct hosts.
    """
    from simpler.worker import RemoteWorkerSpec, Worker, _RunResources  # noqa: PLC0415

    hosts = ("127.0.0.1", "127.0.0.1") if hosts is None else tuple(hosts)
    worker = Worker(level=4, num_sub_workers=0)
    node_ids = tuple(
        worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=f"{host}:{19073 + index}",
                platform=platform,
                device_ids=(0,),
                comm_profile=profile,
                global_device_ranks=(index,),
            )
        )
        for index, host in enumerate(hosts)
    )
    resources = _RunResources()
    worker._worker = object()
    worker._building_run_resources = resources
    worker._control_admission = lambda _api: contextlib.nullcontext()
    return worker, resources, node_ids


def _install_global_domain_failure_injector(monkeypatch, worker, *, fail_phase, fail_node):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415

    calls = []

    def control(worker_id, control_name, payload):
        control_name = ControlName(control_name)
        if control_name is ControlName.COMM_INIT:
            init = decode_comm_init(payload)
            calls.append(("COMM_INIT", worker_id))
            return encode_comm_init_result(
                resolve_global_comm_capability(
                    platform="a2a3sim",
                    profile=init.profile,
                    local_device_count=1,
                )
            )

        assert control_name is ControlName.ALLOC_DOMAIN
        command = decode_domain_command(payload)
        calls.append((command.phase, worker_id))
        if command.phase is fail_phase and worker_id == fail_node:
            raise RuntimeError(f"injected {command.phase.name} failure")
        if command.phase is not GlobalDomainPhase.PREPARE_EXPORT:
            return b""
        descriptors = tuple(
            GlobalDomainDescriptor(
                version=GLOBAL_DOMAIN_VERSION,
                profile_id=GLOBAL_DOMAIN_PROFILE_IDS[command.profile],
                domain_rank=member.domain_rank,
                rank_count=len(command.members),
                mapping_size=4096,
                handle=f"/injected-{member.domain_rank}".encode(),
            )
            for member in command.members
            if member.node_worker_id == worker_id
        )
        return encode_descriptor_table(descriptors)

    monkeypatch.setattr(worker, "_global_domain_control", control)
    return calls


def _close_failure_injection_worker(worker, resources):
    worker._building_run_resources = None
    worker._comm_domains = None
    worker._worker = None
    worker.close()


def _mpi_static_worker():
    from simpler.worker import MpiL3GroupSpec, Worker, _RunResources  # noqa: PLC0415

    worker = Worker(level=4, num_sub_workers=0)
    node_ids = worker.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=("127.0.0.1", "127.0.0.1"),
            platform="a2a3sim",
            device_ids_by_rank=((0,), (0,)),
            comm_profile="sim",
            global_device_ranks_by_rank=((0,), (1,)),
        )
    )
    resources = _RunResources()
    worker._worker = object()
    worker._building_run_resources = resources
    return worker, resources, node_ids


def test_mpi_group_spec_rejects_a_global_device_rank_reused_across_mpi_ranks():
    from simpler.worker import MpiL3GroupSpec  # noqa: PLC0415

    # A global device rank names one device in the cluster, so two mpirun ranks
    # claiming rank 3 is the same defect as one rank listing it twice.
    with pytest.raises(ValueError, match="unique across the whole group"):
        MpiL3GroupSpec(
            hosts=("127.0.0.1", "127.0.0.1"),
            platform="a2a3sim",
            command_port_base=21073,
            health_port_base=22073,
            device_ids_by_rank=((0,), (0,)),
            comm_profile="sim",
            global_device_ranks_by_rank=((3,), (3,)),
        )


@pytest.mark.parametrize(
    "fail_phase",
    (
        GlobalDomainPhase.PREPARE_EXPORT,
        GlobalDomainPhase.IMPORT,
        GlobalDomainPhase.COMMIT,
    ),
)
def test_global_domain_transaction_aborts_all_prepared_nodes_after_phase_failure(monkeypatch, fail_phase):
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker()
    calls = _install_global_domain_failure_injector(
        monkeypatch,
        worker,
        fail_phase=fail_phase,
        fail_node=node_ids[1],
    )
    try:
        with pytest.raises(RuntimeError, match=f"injected {fail_phase.name} failure"):
            worker._allocate_global_domain(
                name="failure-injection",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
                retain_after_run=False,
            )

        abort_nodes = [node_id for phase, node_id in calls if phase is GlobalDomainPhase.ABORT]
        assert abort_nodes == list(node_ids)
        assert worker.live_domains() == {}
        assert not any(r.state == "live" and r.owner for r in worker._domain_registry.records.values())
    finally:
        _close_failure_injection_worker(worker, resources)


def test_allocate_global_domain_builds_one_attachment_row_per_receiver_node(monkeypatch):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker()
    commands = []
    calls = _install_global_domain_failure_injector(
        monkeypatch,
        worker,
        fail_phase=None,
        fail_node=-1,
    )
    original_control = worker._global_domain_control

    def capture_control(worker_id, control_name, payload):
        if ControlName(control_name) is ControlName.ALLOC_DOMAIN:
            commands.append(decode_domain_command(payload))
        return original_control(worker_id, control_name, payload)

    monkeypatch.setattr(worker, "_global_domain_control", capture_control)
    try:
        handle = worker._allocate_global_domain(
            name="attachment-matrix",
            members=((node_ids[0], 0), (node_ids[1], 0)),
            window_size=4096,
            buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
            retain_after_run=False,
        )

        assert len(handle.attachments) == len(node_ids) * len(handle.members)
        for node_worker_id in node_ids:
            row = tuple(attachment for attachment in handle.attachments if attachment.node_worker_id == node_worker_id)
            assert len(row) == len(handle.members)
            assert all(attachment.address_space is AddressSpace.HOST for attachment in row)
        domain_commands = [command for command in commands if command.phase is not GlobalDomainPhase.ABORT]
        assert domain_commands
        prepare_commands = [command for command in domain_commands if command.phase is GlobalDomainPhase.PREPARE_EXPORT]
        later_commands = [
            command for command in domain_commands if command.phase is not GlobalDomainPhase.PREPARE_EXPORT
        ]
        assert prepare_commands
        assert all(command.attachments == handle.attachments for command in prepare_commands)
        assert all(not command.attachments for command in later_commands)
        assert calls
    finally:
        _close_failure_injection_worker(worker, resources)


def _attachment_matrix_for_hosts(monkeypatch, hosts):
    """Allocate one two-rank domain across ``hosts`` and return (node_ids, handle.attachments)."""
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker(hosts=hosts)
    _install_global_domain_failure_injector(monkeypatch, worker, fail_phase=None, fail_node=-1)
    try:
        handle = worker._allocate_global_domain(
            name="attachment-adapters",
            members=((node_ids[0], 0), (node_ids[1], 0)),
            window_size=4096,
            buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
            retain_after_run=False,
        )
        return node_ids, handle.attachments
    finally:
        _close_failure_injection_worker(worker, resources)


def test_attachment_adapters_come_from_the_endpoint_planner_per_pair(monkeypatch):
    # The matrix is per (receiver node, rank) because the answer differs per pair, which is the
    # whole reason a row cannot collapse to one entry per host. Two nodes on one machine reach
    # every rank window the same way; two nodes on different machines reach only their own.
    same_ids, same_host = _attachment_matrix_for_hosts(monkeypatch, ("127.0.0.1", "127.0.0.1"))
    cross_ids, cross_host = _attachment_matrix_for_hosts(monkeypatch, ("10.0.0.1", "10.0.0.2"))

    assert len(same_host) == 4
    assert all(attachment.adapter_kind is AdapterKind.OWNER_DELEGATED_COPY for attachment in same_host)
    assert all(attachment.adapter_profile is AdapterProfile.HOST_VMM_COPY for attachment in same_host)

    # Cross-machine: the diagonal (a node reaching the rank it owns) resolves; the off-diagonal
    # does not, and is carried as an adapter-less host consumer rather than as a usable mapping.
    assert len(cross_host) == 4
    resolved = {index for index, attachment in enumerate(cross_host) if attachment.adapter_kind is not None}
    assert resolved == {0, 3}, cross_host
    for index in (1, 2):
        assert cross_host[index].adapter_kind is None
        assert cross_host[index].adapter_profile is None
    # An adapter-less row is still a complete, HOST-consumer record on both topologies.
    for row in (same_host, cross_host):
        assert all(attachment.address_space is AddressSpace.HOST for attachment in row)
        assert all(attachment.role is AttachmentRole.CONSUMER for attachment in row)
    assert same_ids == cross_ids


def test_attachment_adapter_pair_survives_a_wire_round_trip_with_and_without_an_adapter(monkeypatch):
    # The None adapter has to survive encode/decode as None, not as a zero-valued enumerator:
    # the wire reserves id 0 for "no adapter", and only a round trip proves the two directions
    # agree on that.
    _node_ids, cross_host = _attachment_matrix_for_hosts(monkeypatch, ("10.0.0.1", "10.0.0.2"))
    assert any(attachment.adapter_kind is None for attachment in cross_host)
    assert any(attachment.adapter_kind is not None for attachment in cross_host)

    command = _domain_command(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=7,
        generation=1,
        name="round-trip",
        profile="sim",
        window_size=4096,
        members=_members(),
        buffers=(GlobalDomainBuffer("payload", 4096),),
        attachments=cross_host,
    )
    assert decode_domain_command(encode_domain_command(command)).attachments == cross_host


def test_global_domain_abort_failure_preserves_primary_error_and_poisons_admission(monkeypatch):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker()
    calls = _install_global_domain_failure_injector(
        monkeypatch,
        worker,
        fail_phase=GlobalDomainPhase.IMPORT,
        fail_node=node_ids[1],
    )
    original_control = worker._global_domain_control

    def fail_first_abort(worker_id, control_name, payload):
        if ControlName(control_name) is ControlName.ALLOC_DOMAIN:
            command = decode_domain_command(payload)
            if command.phase is GlobalDomainPhase.ABORT and worker_id == node_ids[0]:
                calls.append((command.phase, worker_id))
                raise RuntimeError("injected ABORT failure")
        return original_control(worker_id, control_name, payload)

    monkeypatch.setattr(worker, "_global_domain_control", fail_first_abort)
    try:
        with pytest.raises(RuntimeError, match="injected IMPORT failure"):
            worker._allocate_global_domain(
                name="abort-failure-injection",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
                retain_after_run=False,
            )

        abort_nodes = [node_id for phase, node_id in calls if phase is GlobalDomainPhase.ABORT]
        assert abort_nodes == list(node_ids)
        # Rollback debt stays registered and prevents further admission.
        leaked = worker._ordered_cleanup_error
        assert leaked is not None
        assert "abort-failure-injection" in str(leaked)
        assert f"node worker {node_ids[0]}" in str(leaked.__cause__)
        assert isinstance(leaked.__cause__, RuntimeError)
        assert "injected ABORT failure" in str(leaked.__cause__)
        assert worker.live_domains() == {}
        assert not any(r.state == "live" and r.owner for r in worker._domain_registry.records.values())
    finally:
        _close_failure_injection_worker(worker, resources)


def test_mixed_local_remote_import_failure_rolls_back_local_node(monkeypatch):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415
    from simpler.worker import (  # noqa: PLC0415
        CTRL_GLOBAL_DOMAIN_IMPORT,
        CTRL_GLOBAL_DOMAIN_PREPARE,
        CTRL_GLOBAL_DOMAIN_RELEASE,
        LOCAL_DOMAIN_MAGIC,
        LOCAL_IMPORT_REPLY,
        LOCAL_IMPORT_REQUEST,
        LOCAL_PREPARE_REPLY,
        LOCAL_PREPARE_REQUEST,
        RemoteWorkerSpec,
        Worker,
        _run_local_global_domain_control,
        _RunResources,
    )

    released_local_workers = []

    class FakeLocalNativeWorker:
        def control_payload(self, _kind, worker_id, control, payload, _timeout):
            if control == CTRL_GLOBAL_DOMAIN_PREPARE:
                (
                    _magic,
                    _version,
                    domain_id,
                    generation,
                    domain_rank,
                    rank_count,
                    profile_id,
                    window_size,
                ) = LOCAL_PREPARE_REQUEST.unpack_from(payload, 0)
                descriptor = GlobalDomainDescriptor(
                    version=GLOBAL_DOMAIN_VERSION,
                    profile_id=profile_id,
                    domain_rank=domain_rank,
                    rank_count=rank_count,
                    mapping_size=window_size,
                    handle=b"/mixed-local",
                )
                return (
                    LOCAL_PREPARE_REPLY.pack(
                        LOCAL_DOMAIN_MAGIC,
                        GLOBAL_DOMAIN_VERSION,
                        domain_id,
                        generation,
                        0x1000,
                        window_size,
                    )
                    + descriptor.encode()
                )
            if control == CTRL_GLOBAL_DOMAIN_IMPORT:
                _magic, _version, domain_id, generation, _rank_count = LOCAL_IMPORT_REQUEST.unpack_from(payload, 0)
                return LOCAL_IMPORT_REPLY.pack(
                    LOCAL_DOMAIN_MAGIC,
                    GLOBAL_DOMAIN_VERSION,
                    domain_id,
                    generation,
                    0x2000,
                    0x1000,
                    4096,
                )
            assert control == CTRL_GLOBAL_DOMAIN_RELEASE
            released_local_workers.append(worker_id)
            return b""

    parent = Worker(level=4, num_sub_workers=0)
    local = Worker(
        level=3,
        device_ids=(0,),
        num_sub_workers=0,
        platform="a2a3sim",
        comm_profile="sim",
        global_device_ranks=(0,),
    )
    local_node_id = parent.add_worker(local)
    remote_node_id = parent.add_remote_worker(
        RemoteWorkerSpec(
            endpoint="127.0.0.1:19073",
            platform="a2a3sim",
            device_ids=(0,),
            comm_profile="sim",
            global_device_ranks=(1,),
        )
    )
    resources = _RunResources()
    parent._worker = object()
    parent._building_run_resources = resources
    local._worker = FakeLocalNativeWorker()
    local_runtime = parent._resolved_global_nodes()[local_node_id]
    local_comm_inits = {}
    calls = []

    def control(worker_id, control_name, payload):
        control_name = ControlName(control_name)
        if worker_id == local_node_id:
            event = decode_domain_command(payload).phase if control_name is ControlName.ALLOC_DOMAIN else control_name
            calls.append(("local", event))
            return _run_local_global_domain_control(
                local,
                local_runtime,
                local_comm_inits,
                control_name,
                payload,
            )

        assert worker_id == remote_node_id
        if control_name is ControlName.COMM_INIT:
            command = decode_comm_init(payload)
            calls.append(("remote", control_name))
            return encode_comm_init_result(
                resolve_global_comm_capability(
                    platform="a2a3sim",
                    profile=command.profile,
                    local_device_count=1,
                )
            )

        command = decode_domain_command(payload)
        calls.append(("remote", command.phase))
        if command.phase is GlobalDomainPhase.PREPARE_EXPORT:
            descriptor = GlobalDomainDescriptor(
                version=GLOBAL_DOMAIN_VERSION,
                profile_id=GLOBAL_DOMAIN_PROFILE_IDS[command.profile],
                domain_rank=1,
                rank_count=2,
                mapping_size=4096,
                handle=b"/mixed-remote",
            )
            return encode_descriptor_table((descriptor,))
        if command.phase is GlobalDomainPhase.IMPORT:
            raise RuntimeError("remote import failed")
        return b""

    monkeypatch.setattr(parent, "_global_domain_control", control)
    try:
        with pytest.raises(RuntimeError, match="remote import failed"):
            parent._allocate_global_domain(
                name="mixed-rollback",
                members=((local_node_id, 0), (remote_node_id, 0)),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 64, 64)],
                retain_after_run=False,
            )

        assert released_local_workers == [0]
        assert local._global_node_domains == {}
        assert ("local", GlobalDomainPhase.ABORT) in calls
        assert ("remote", GlobalDomainPhase.ABORT) in calls
        assert parent.live_domains() == {}
        assert not any(r.state == "live" and r.owner for r in parent._domain_registry.records.values())
    finally:
        parent._building_run_resources = None
        parent._worker = None
        local._global_node_domains.clear()
        local._worker = None
        parent.close()


def test_allocate_global_domain_rejects_unsupported_capability_before_control(monkeypatch):
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _failure_injection_worker(platform="a5", profile="sim")
    calls = []
    monkeypatch.setattr(worker, "_global_domain_control", lambda *args: calls.append(args))
    try:
        with pytest.raises(ValueError, match="Global CommDomain is not supported"):
            worker._allocate_global_domain(
                name="unsupported",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
                retain_after_run=False,
            )

        assert calls == []
        assert worker.live_domains() == {}
        assert not any(r.state == "live" and r.owner for r in worker._domain_registry.records.values())
    finally:
        _close_failure_injection_worker(worker, resources)


def test_global_domain_descriptor_table_rejects_different_mapping_sizes():
    descriptors = list(_descriptors())
    descriptors[1] = GlobalDomainDescriptor(
        version=GLOBAL_DOMAIN_VERSION,
        profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
        domain_rank=1,
        rank_count=2,
        mapping_size=8192,
        handle=b"/simpler-test-1",
    )

    with pytest.raises(ValueError, match="mapping sizes differ"):
        validate_descriptor_table(tuple(descriptors), rank_count=2, profile="sim")


def _local_node_domain_commands():
    members = (
        GlobalDomainMember(0, 0, 0, 0),
        GlobalDomainMember(0, 1, 1, 1),
    )
    descriptors = tuple(
        GlobalDomainDescriptor(
            version=GLOBAL_DOMAIN_VERSION,
            profile_id=GLOBAL_DOMAIN_PROFILE_IDS["sim"],
            domain_rank=rank,
            rank_count=2,
            mapping_size=4096,
            handle=f"/local-node-{rank}".encode(),
        )
        for rank in range(2)
    )
    common = {
        "domain_id": 29,
        "generation": 3,
        "name": "local-node",
        "profile": "sim",
        "window_size": 4096,
        "members": members,
        "buffers": (GlobalDomainBuffer("payload", 64),),
    }
    prepare = _domain_command(phase=GlobalDomainPhase.PREPARE_EXPORT, attachments=_attachments()[:2], **common)
    imported = _domain_command(phase=GlobalDomainPhase.IMPORT, descriptors=descriptors, **common)
    return prepare, imported


@pytest.mark.parametrize("release_fails", [False, True], ids=["rollback-succeeds", "rollback-fails"])
def test_node_abort_result_preserves_owner_cleanup_obligation(release_fails):
    from simpler.worker import (  # noqa: PLC0415
        Worker,
        _GlobalNodeDomainState,
        _GlobalNodeRuntime,
        _run_local_global_domain_control,
    )

    prepare, _ = _local_node_domain_commands()
    _, _, config, _ = decode_domain_metadata(prepare.domain_metadata)
    owner, node = Worker(level=4, num_sub_workers=0), Worker(level=3, device_ids=(0, 1), num_sub_workers=0)
    captured, releases = [], []
    runtime = _GlobalNodeRuntime(0, (0, 1), "a2a3sim", "sim", (0, 1), 0, 1, "test", False)

    class NativeNode:
        def control_payload(self, _kind, chip, _control, _payload, _timeout):
            releases.append(chip)
            if release_fails:
                raise RuntimeError("chip release failed")
            return b""

    node._worker = NativeNode()

    def allocate(record):
        captured.append(record)
        command = replace(
            prepare,
            name=record.name,
            domain_id=record.ref.domain_id,
            generation=record.ref.generation,
            domain_metadata=encode_domain_metadata(record),
        )
        view = node._domain_registry.import_domain(command.domain_metadata).handle
        node._global_node_domains[command.key] = _GlobalNodeDomainState(
            command=command,
            view=view,
            allocation_id=1,
            prepared_domain_ranks={0, 1},
            phase=GlobalDomainPhase.IMPORT,
        )
        record.backend = {"kind": "global", "nodes": [0], "mpi_group": None, "committed": False, "command": command}
        owner._global_domain_control = lambda _node, control, payload: _run_local_global_domain_control(
            node,
            runtime,
            {"tp": GlobalCommInitCommand("test", "topology", command.profile, 0, 1, command.members)},
            control,
            payload,
        )
        raise RuntimeError("original import failure")

    owner._domain_registry._allocate = allocate
    try:
        with pytest.raises(RuntimeError, match="original import failure"):
            owner._domain_registry.allocate("tp", config)
        (record,) = captured
        assert releases == [0, 1]
        if release_fails:
            assert record.state == "released" and record.free_error is not None
            assert record.backend["nodes"] == [0]
            assert owner._ordered_cleanup_error is not None and node._ordered_cleanup_error is not None
            assert record.ref in owner._domain_registry.records
            with pytest.raises(RuntimeError, match="chip release failed"):
                owner._destroy_global_domain_record(record)
            assert releases == [0, 1], "failed node free must not be replayed"
        else:
            assert record.state == "freed" and record.free_error is None
            assert record.backend["nodes"] == [] and node._global_node_domains == {}
            assert owner._ordered_cleanup_error is None and node._ordered_cleanup_error is None
            assert node._domain_registry.records == {}
    finally:
        for worker in (owner, node):
            worker._worker = None
            worker._comm_domains = None
            worker._global_node_domains.clear()
            worker.close()


def test_node_import_failure_rolls_back_partial_local_ranks_and_uses_configured_timeout():
    from simpler.worker import (  # noqa: PLC0415
        CTRL_GLOBAL_DOMAIN_IMPORT,
        CTRL_GLOBAL_DOMAIN_RELEASE,
        LOCAL_DOMAIN_MAGIC,
        LOCAL_IMPORT_REPLY,
        Worker,
    )

    prepare, imported = _local_node_domain_commands()
    release_workers = []
    timeouts = []

    class FakeNativeWorker:
        def control_payload(self, _kind, worker_id, control, _payload, timeout):
            timeouts.append(timeout)
            if control == CTRL_GLOBAL_DOMAIN_IMPORT:
                if worker_id == 1:
                    raise RuntimeError("second local import failed")
                return LOCAL_IMPORT_REPLY.pack(
                    LOCAL_DOMAIN_MAGIC,
                    GLOBAL_DOMAIN_VERSION,
                    imported.domain_id,
                    imported.generation,
                    0x1000,
                    0x2000,
                    4096,
                )
            assert control == CTRL_GLOBAL_DOMAIN_RELEASE
            release_workers.append(worker_id)
            return b""

    worker = Worker(level=3, device_ids=(0, 1), py_control_timeout_s=7.25)
    worker._worker = FakeNativeWorker()
    _install_prepared_node(worker, prepare)
    try:
        with pytest.raises(RuntimeError, match="second local import failed"):
            worker._import_global_domain_node(imported, 0)

        assert release_workers == [0, 1]
        assert worker._global_node_domains == {}
        assert timeouts == [7.25, 7.25, 7.25, 7.25]
    finally:
        worker._global_node_domains.clear()
        worker._worker = None
        worker.close()


def test_node_import_reuses_prepared_attachment_row_for_view():
    from simpler.worker import (  # noqa: PLC0415
        CTRL_GLOBAL_DOMAIN_IMPORT,
        CTRL_GLOBAL_DOMAIN_RELEASE,
        LOCAL_DOMAIN_MAGIC,
        LOCAL_IMPORT_REPLY,
        Worker,
    )

    prepare, imported = _local_node_domain_commands()

    class FakeNativeWorker:
        def control_payload(self, _kind, _worker_id, control, _payload, _timeout):
            if control == CTRL_GLOBAL_DOMAIN_IMPORT:
                return LOCAL_IMPORT_REPLY.pack(
                    LOCAL_DOMAIN_MAGIC,
                    GLOBAL_DOMAIN_VERSION,
                    imported.domain_id,
                    imported.generation,
                    0x1000,
                    0x2000,
                    4096,
                )
            assert control == CTRL_GLOBAL_DOMAIN_RELEASE
            return b""

    worker = Worker(level=3, device_ids=(0, 1))
    worker._worker = FakeNativeWorker()
    _install_prepared_node(worker, prepare)
    commit = _domain_command(
        phase=GlobalDomainPhase.COMMIT,
        domain_id=imported.domain_id,
        generation=imported.generation,
        name=imported.name,
        profile=imported.profile,
        window_size=imported.window_size,
        members=imported.members,
        buffers=imported.buffers,
        descriptors=imported.descriptors,
    )
    try:
        worker._import_global_domain_node(imported, 0)
        state = worker._global_node_domains[prepare.key]
        assert state.command.attachments == prepare.attachments
        assert state.view is not None
        assert state.view.attachments == prepare.attachments_for_node(0)

        worker._commit_global_domain_node(commit)
        assert state.view.committed
    finally:
        worker._release_global_domain_node(_release_command(prepare.domain_id, prepare.generation))
        worker._worker = None
        worker.close()


def test_node_release_invalidates_committed_view_before_partial_fanout_failure():
    from simpler.worker import (  # noqa: PLC0415
        CTRL_GLOBAL_DOMAIN_RELEASE,
        Worker,
        _GlobalNodeDomainState,
    )

    prepare, imported = _local_node_domain_commands()
    timeouts = []

    class FakeNativeWorker:
        fail_second = True

        def control_payload(self, _kind, worker_id, control, _payload, timeout):
            assert control == CTRL_GLOBAL_DOMAIN_RELEASE
            timeouts.append(timeout)
            if self.fail_second and worker_id == 1:
                raise RuntimeError("partial release failed")
            return b""

    native = FakeNativeWorker()
    worker = Worker(level=3, device_ids=(0, 1), py_control_timeout_s=6.5)
    worker._worker = native
    view = worker._domain_registry.import_domain(imported.domain_metadata).handle
    view._record.state = "live"
    view._record.mapping_size = 4096
    state = _GlobalNodeDomainState(
        command=imported,
        allocation_id=imported.domain_id,
        prepared_domain_ranks={0, 1},
        view=view,
        phase=GlobalDomainPhase.COMMIT,
    )
    worker._global_node_domains[imported.key] = state
    try:
        with pytest.raises(RuntimeError, match="partial release failed"):
            worker._release_global_domain_node(_release_command(imported.domain_id, imported.generation))

        assert state.phase is GlobalDomainPhase.ABORT
        assert not view.committed
        with pytest.raises(KeyError, match="not committed"):
            worker._get_global_domain(imported.domain_id)
        with pytest.raises(RuntimeError, match="released"):
            view[0]

        native.fail_second = False
        with pytest.raises(RuntimeError, match="partial release failed"):
            worker._release_global_domain_node(_release_command(imported.domain_id, imported.generation))
        assert worker._global_node_domains[imported.key] is state
        assert state.view._record.free_error is not None
        assert worker._ordered_cleanup_error is not None
        assert not state.view.freed and timeouts == [6.5, 6.5]
    finally:
        worker._global_node_domains.clear()
        worker._domain_registry.context_torn_down()
        worker._worker = None
        worker.close()


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tcp_ports(ports: tuple[int, ...], timeout_s: float = 5.0) -> None:
    pending = set(ports)
    deadline = time.monotonic() + timeout_s
    while pending:
        for port in tuple(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"remote L3 daemons did not become ready on ports {sorted(pending)}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=min(0.1, remaining)):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.01)


@pytest.mark.skipif(os.name == "nt", reason="hierarchical workers require fork")
def test_two_remote_daemons_build_and_copy_global_domain_without_mpirun():
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415
    from simpler.worker import RemoteWorkerSpec, Worker  # noqa: PLC0415

    ports = (_free_tcp_port(), _free_tcp_port())
    daemons = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "simpler.remote_l3_worker",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for port in ports
    ]
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=20)
    captured: dict[str, object] = {}
    try:
        _wait_for_tcp_ports(ports)
        node_ids = tuple(
            worker.add_remote_worker(
                RemoteWorkerSpec(
                    endpoint=f"127.0.0.1:{port}",
                    platform="a2a3sim",
                    device_ids=(0,),
                    comm_profile="sim",
                )
            )
            for port in ports
        )
        worker.init()

        def parent_orch(orch, _args, _cfg):
            domain = orch.allocate_global_domain(
                name="tcp-global",
                members=((node_ids[0], 0), (node_ids[1], 0)),
                window_size=4096,
                buffers=(CommBufferSpec("payload", "uint8", 64, 64),),
                retain_after_run=True,
            )
            orch.copy_to_global_domain(domain, 0, b"node-zero", buffer="payload")
            orch.copy_to_global_domain(domain, 1, b"node-one", buffer="payload")
            captured["ranks"] = tuple(member.global_device_rank for member in domain.members)
            captured["handle"] = domain

        worker.run(parent_orch)
        assert not captured["handle"].freed

        def read_orch(orch, _args, _cfg):
            domain = captured["handle"]
            try:
                captured["rank0"] = orch.copy_from_global_domain(domain, 0, len(b"node-zero"), buffer="payload")
                captured["rank1"] = orch.copy_from_global_domain(domain, 1, len(b"node-one"), buffer="payload")
            finally:
                domain.release()

        worker.run(read_orch)
        assert captured["rank0"] == b"node-zero"
        assert captured["rank1"] == b"node-one"
        assert captured["ranks"] == (0, 1)
        assert captured["handle"].freed
    finally:
        worker.close()
        for daemon in daemons:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="hierarchical workers require fork")
def test_local_and_remote_l3_build_and_copy_global_domain_without_mpirun():
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415
    from simpler.worker import RemoteWorkerSpec, Worker  # noqa: PLC0415

    port = _free_tcp_port()
    daemon = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "simpler.remote_l3_worker",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=20)
    captured: dict[str, object] = {}
    try:
        _wait_for_tcp_ports((port,))
        local_node_id = worker.add_worker(
            Worker(
                level=3,
                device_ids=[0],
                num_sub_workers=0,
                platform="a2a3sim",
                runtime="tensormap_and_ringbuffer",
                comm_profile="sim",
                global_device_ranks=(0,),
            )
        )
        remote_node_id = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=f"127.0.0.1:{port}",
                platform="a2a3sim",
                device_ids=(0,),
                comm_profile="sim",
                global_device_ranks=(1,),
            )
        )
        worker.init()

        def build_orch(orch, _args, _cfg):
            domain = orch.allocate_global_domain(
                name="mixed-global",
                members=((local_node_id, 0), (remote_node_id, 0)),
                window_size=4096,
                buffers=(CommBufferSpec("payload", "uint8", 64, 64),),
                retain_after_run=True,
            )
            orch.copy_to_global_domain(domain, 0, b"local-l3", buffer="payload")
            orch.copy_to_global_domain(domain, 1, b"remote-l3", buffer="payload")
            captured["ranks"] = tuple(member.global_device_rank for member in domain.members)
            captured["domain"] = domain

        worker.run(build_orch)
        domain = captured["domain"]
        assert not domain.freed

        def read_orch(orch, _args, _cfg):
            try:
                captured["local"] = orch.copy_from_global_domain(
                    domain,
                    0,
                    len(b"local-l3"),
                    buffer="payload",
                )
                captured["remote"] = orch.copy_from_global_domain(
                    domain,
                    1,
                    len(b"remote-l3"),
                    buffer="payload",
                )
            finally:
                domain.release()

        worker.run(read_orch)
        assert captured["local"] == b"local-l3"
        assert captured["remote"] == b"remote-l3"
        assert captured["ranks"] == (0, 1)
        assert domain.freed
    finally:
        worker.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)


def test_global_domain_control_ids_do_not_overlap_worker_controls():
    from simpler.worker import _CTRL_COMMITTED_DEVICE_MEMORY, _CTRL_GLOBAL_DOMAIN_NODE  # noqa: PLC0415

    control_ids = (
        _CTRL_COMMITTED_DEVICE_MEMORY,
        CTRL_GLOBAL_DOMAIN_PREPARE,
        CTRL_GLOBAL_DOMAIN_IMPORT,
        CTRL_GLOBAL_DOMAIN_RELEASE,
        CTRL_GLOBAL_DOMAIN_COPY_TO,
        CTRL_GLOBAL_DOMAIN_COPY_FROM,
        _CTRL_GLOBAL_DOMAIN_NODE,
    )

    assert len(control_ids) == len(set(control_ids))


def test_mpirun_group_global_domain_uses_mpi_prepare_commit_without_l4_import(monkeypatch):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    worker, resources, node_ids = _mpi_static_worker()
    calls = []
    commands = []

    def control(worker_id, control_name, payload, *, group=False):
        control_name = ControlName(control_name)
        if control_name is ControlName.COMM_INIT:
            init = decode_comm_init(payload)
            calls.append(("COMM_INIT", worker_id))
            return encode_comm_init_result(
                resolve_global_comm_capability(
                    platform="a2a3sim",
                    profile=init.profile,
                    local_device_count=1,
                )
            )
        assert control_name is ControlName.ALLOC_DOMAIN
        command = decode_domain_command(payload)
        commands.append(command)
        calls.append((command.phase, worker_id))
        if command.phase is GlobalDomainPhase.PREPARE_EXPORT:
            descriptors = tuple(
                GlobalDomainDescriptor(
                    version=GLOBAL_DOMAIN_VERSION,
                    profile_id=GLOBAL_DOMAIN_PROFILE_IDS[command.profile],
                    domain_rank=member.domain_rank,
                    rank_count=len(command.members),
                    mapping_size=4096,
                    handle=f"/mpi-prepared-{member.domain_rank}".encode(),
                )
                for member in command.members
            )
            return encode_descriptor_table(descriptors)
        if command.phase is GlobalDomainPhase.IMPORT:
            raise RuntimeError("L4 broker IMPORT should not run for a full mpirun group")
        return b""

    monkeypatch.setattr(worker, "_global_domain_control", control)
    try:
        handle = worker._allocate_global_domain(
            name="mpi-static",
            members=((node_ids[0], 0), (node_ids[1], 0)),
            window_size=4096,
            buffers=[CommBufferSpec("payload", "uint8", 4096, 4096)],
            retain_after_run=False,
        )

        assert handle.mapping_size == 4096
        assert handle.members[0].global_device_rank == 0
        assert handle.members[1].global_device_rank == 1
        counts = Counter(phase for phase, _worker_id in calls)
        assert counts["COMM_INIT"] == 2
        assert counts[GlobalDomainPhase.PREPARE_EXPORT] == 1
        assert counts[GlobalDomainPhase.COMMIT] == 1
        assert counts[GlobalDomainPhase.IMPORT] == 0
        assert len(commands) == 2
        prepare_command = next(command for command in commands if command.phase is GlobalDomainPhase.PREPARE_EXPORT)
        commit_command = next(command for command in commands if command.phase is GlobalDomainPhase.COMMIT)
        assert prepare_command.attachments
        assert not commit_command.attachments
        assert len(prepare_command.attachments_for_node(node_ids[0])) == len(prepare_command.members)
        assert len(prepare_command.attachments_for_node(node_ids[1])) == len(prepare_command.members)
        group_phases = [
            worker_id
            for phase, worker_id in calls
            if phase in (GlobalDomainPhase.PREPARE_EXPORT, GlobalDomainPhase.COMMIT)
        ]
        assert group_phases == [node_ids[0], node_ids[0]]
        assert worker.live_domains()["mpi-static"] is handle
        assert resources.domain_use_token in handle._record.uses
    finally:
        _close_failure_injection_worker(worker, resources)


def test_mpi_global_domain_collective_timeout_releases_local_state():
    from simpler.mpi_l3_session import MpiGlobalDomainExchange  # noqa: PLC0415

    stuck = threading.Event()

    class _Comm:
        aborted = False

        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def allgather(_payload):
            # A peer rank never entering the collective looks like this to
            # the pickle-based blocking allgather.
            stuck.wait(30.0)

        def Abort(self, _error_code):
            self.aborted = True
            raise RuntimeError("fake MPI abort")

    comm = _Comm()
    exchange = MpiGlobalDomainExchange(comm, group_worker_ids=(7,), timeout_s=0.05)
    releases = []

    try:
        with pytest.raises(TimeoutError, match="prepare timed out"):
            exchange._allgather(b"payload", operation="prepare", on_timeout=lambda: releases.append(True))
    finally:
        stuck.set()

    assert releases == [True]
    assert comm.aborted


def test_mpi_global_domain_collective_uses_pickle_allgather():
    from simpler.mpi_l3_session import MpiGlobalDomainExchange  # noqa: PLC0415

    class _Comm:
        payloads = []

        @staticmethod
        def Get_rank():
            return 0

        def allgather(self, payload):
            self.payloads.append(payload)
            return [payload, b"peer"]

    comm = _Comm()
    exchange = MpiGlobalDomainExchange(comm, group_worker_ids=(7,), timeout_s=1.0)

    gathered = exchange._allgather(
        b"payload",
        operation="prepare",
        on_timeout=lambda: pytest.fail("pickle-based allgather completed without timing out"),
    )

    assert gathered == [b"payload", b"peer"]
    assert comm.payloads == [b"payload"]


def test_mpi_global_domain_prepare_failure_releases_before_collective():
    from simpler.mpi_l3_session import MpiGlobalDomainExchange  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    class _Comm:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def allgather(payload):
            return [payload]

    class _InnerWorker:
        released = False

        @staticmethod
        def _prepare_global_domain_node(_command, _worker_id):
            raise RuntimeError("injected prepare failure")

        def _release_global_domain_node(self, _command, *, suppress_errors):
            assert suppress_errors
            self.released = True

    command = _domain_command(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=20,
        generation=1,
        name="mpi-failure",
        profile="sim",
        window_size=4096,
        members=(GlobalDomainMember(7, 0, 0, 0),),
        buffers=(),
    )
    inner_worker = _InnerWorker()
    exchange = MpiGlobalDomainExchange(_Comm(), group_worker_ids=(7,), timeout_s=1.0)

    with pytest.raises(RuntimeError, match="prepare failed on rank 0"):
        exchange.prepare_import(command, cast(Worker, inner_worker), 7)

    assert inner_worker.released


def test_mpirun_group_cleanup_continues_after_one_process_wait_fails():
    class _Process:
        def __init__(self, *, fail_wait):
            self.fail_wait = fail_wait
            self.waited = False

        @staticmethod
        def poll():
            return 0

        def wait(self, *, timeout):
            assert timeout == 0.1
            self.waited = True
            if self.fail_wait:
                raise RuntimeError("injected wait failure")
            return 0

    worker, resources, _node_ids = _mpi_static_worker()
    group = worker._mpi_l3_groups[0]
    first_process = _Process(fail_wait=True)
    second_process = _Process(fail_wait=False)
    first = type(group)(
        group_id="first",
        spec=group.spec,
        ranks=group.ranks,
        process=cast(subprocess.Popen, first_process),
    )
    second = type(group)(
        group_id="second",
        spec=group.spec,
        ranks=group.ranks,
        process=cast(subprocess.Popen, second_process),
    )
    worker._mpi_l3_groups[:] = [first, second]
    try:
        with pytest.raises(RuntimeError, match="first cleanup wait after terminate"):
            worker._close_mpirun_groups(timeout_s=0.1)

        assert first_process.waited
        assert second_process.waited
        assert first.process is None
        assert second.process is None
    finally:
        worker._mpi_l3_groups.clear()
        _close_failure_injection_worker(worker, resources)


@pytest.mark.parametrize("field", ["domain_id", "generation", "name", "window_size", "members", "buffers"])
def test_domain_command_rejects_metadata_that_disagrees_with_outer_command(field):
    command = _domain_command(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=11,
        generation=1,
        name="tp",
        profile="sim",
        window_size=2048,
        members=_members(),
        buffers=(GlobalDomainBuffer("payload", 128),),
        attachments=_attachments(),
    )
    changes = dict(
        domain_id=12,
        generation=2,
        name="other",
        window_size=4096,
        members=tuple(replace(member, global_device_rank=member.global_device_rank + 10) for member in command.members),
        buffers=(GlobalDomainBuffer("payload", 64),),
    )
    with pytest.raises(ValueError, match="metadata"):
        encode_domain_command(replace(command, **{field: changes[field]}))
