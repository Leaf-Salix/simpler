# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import struct
from dataclasses import replace

import pytest
from simpler import comm_endpoints as ce
from simpler.buffer import AddressSpace
from simpler.comm_domain import HostWindowSpec
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_MAX_ATTACHMENTS,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainAttachment,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainHostConsumerAttachment,
    GlobalDomainHostWindow,
    GlobalDomainMember,
    GlobalDomainPhase,
    decode_domain_command,
    encode_domain_command,
)


def _grant():
    return GlobalDomainHostConsumerAttachment(
        node_worker_id=0,
        sub_worker_id=1,
        consumer=ce.EndpointIdentity(b"creator1", 0, 3),
        provider_ranks=(1,),
    )


def _command(*grants):
    return GlobalDomainCommand(
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        domain_id=7,
        generation=3,
        name="mixed",
        profile="sim",
        window_size=4096,
        members=(GlobalDomainMember(0, 0, 0, 0), GlobalDomainMember(1, 0, 1, 1)),
        buffers=(GlobalDomainBuffer("payload", 64),),
        owner_instance_id=b"creator1",
        host_windows=(
            GlobalDomainHostWindow(0, HostWindowSpec(0, 128), 12),
            GlobalDomainHostWindow(1, HostWindowSpec(0, 128), 13),
        ),
        attachments=tuple(
            GlobalDomainAttachment(
                node,
                AddressSpace.HOST,
                ce.AttachmentRole.CONSUMER,
                ce.AdapterKind.OWNER_DELEGATED_COPY if node == rank else ce.AdapterKind.EXPLICIT_TRANSFER,
                ce.AdapterProfile.HOST_VMM_COPY if node == rank else ce.AdapterProfile.REMOTE_COPY,
            )
            for node in (0, 1)
            for rank in (0, 1)
        ),
        host_consumer_attachments=tuple(grants),
    )


def test_sub_endpoint_resolves_in_existing_registry_and_preserves_l_path_keys():
    paths = ("L4", "L4/L3[0]", "L4/L3[0]/L2[0]", "L4/L3[1]")
    assert [ce.parse_endpoint_path(path, root_level=4).sort_key for path in paths] == [
        ((4, -1),),
        ((4, -1), (3, 0)),
        ((4, -1), (3, 0), (2, 0)),
        ((4, -1), (3, 1)),
    ]
    sub = "L4/L3[0]/SUB[1]"
    registry = ce.EndpointRegistry.from_snapshot(
        ce._EndpointTopologySnapshot(
            4,
            b"creator1",
            tuple(ce._EndpointTopologyEntry(path, ce.HOST_CPU, "local") for path in (*paths[:2], sub)),
        ),
        registry_epoch=0,
    )
    assert registry.resolve_members((ce.at(sub, ce.HOST_CPU),))[0].path == sub
    assert [record.path for record in registry.resolve_members((ce.under("L4/L3[0]", ce.HOST_CPU),))] == [sub]
    assert ce.parse_endpoint_path(sub, root_level=4).segments[-1] == ce.EndpointPathSegment(None, 1)


@pytest.mark.parametrize("path", ["SUB[0]", "L4/SUB[0]/L2[0]", "L4/SUB[0]/SUB[1]", "L4/SUB", "L4/SUB[-1]", "L3/SUB[0]"])
def test_sub_path_rejects_invalid_placement_or_index(path):
    with pytest.raises(ValueError):
        ce.parse_endpoint_path(path, root_level=4)


@pytest.mark.parametrize("deployment", [ce.DEVICE_AICORE, ce.DEVICE_AICPU])
def test_sub_registry_rejects_device_deployment(deployment):
    with pytest.raises(ValueError, match="HOST_CPU"):
        ce.EndpointRegistry.from_snapshot(
            ce._EndpointTopologySnapshot(
                4, b"creator1", (ce._EndpointTopologyEntry("L4/L3[0]/SUB[1]", deployment, "local"),)
            ),
            registry_epoch=0,
        )


def test_host_consumer_roundtrip_preserves_canonical_plan_and_native_version():
    first = replace(_grant(), provider_ranks=(0, 1))
    second = replace(first, sub_worker_id=2, consumer=replace(first.consumer, endpoint_id=4), provider_ranks=(1,))
    command = _command(first, second)
    encoded = encode_domain_command(command)
    assert encoded[:4] == b"SDC3"
    assert GLOBAL_DOMAIN_VERSION == 3
    assert decode_domain_command(encoded) == command
    assert decode_domain_command(encode_domain_command(_command())).host_consumer_attachments == ()


def test_two_local_l3_workers_keep_same_node_planner_grants_in_codec():
    from simpler.worker import Worker, _Lifecycle  # noqa: PLC0415

    root = Worker(level=4, num_sub_workers=0)
    for device in (0, 1):
        root.add_worker(Worker(level=3, device_ids=[device], num_sub_workers=1, comm_profile="sim"))
    root._lifecycle = _Lifecycle.READY
    command = _command()
    command = replace(
        command,
        attachments=root._global_domain_attachment_matrix(
            command.members, (0, 1), command.window_size, profile=command.profile, host_nodes=(0, 1)
        ),
    )
    grants = root._global_host_consumer_attachments(command)
    cross_l3 = next(grant for grant in grants if grant.node_worker_id == 0)
    assert 1 in cross_l3.provider_ranks
    registry = root._get_endpoint_registry()
    provider = registry.resolve_members((ce.at("L4/L3[1]/L2[0]", ce.DEVICE_AICORE),))[0]
    assert registry.same_node(provider.identity, cross_l3.consumer)
    assert command.attachments_for_node(0)[1].adapter_profile is ce.AdapterProfile.HOST_VMM_COPY
    command = replace(command, owner_instance_id=root._owner_instance_id, host_consumer_attachments=grants)
    assert decode_domain_command(encode_domain_command(command)) == command


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_worker_id", -1),
        ("node_worker_id", 2),
        ("node_worker_id", 2**31),
        ("sub_worker_id", -1),
        ("sub_worker_id", 2**32),
        ("sub_worker_id", True),
        ("provider_ranks", (-1,)),
        ("provider_ranks", (2,)),
        ("provider_ranks", (True,)),
        ("provider_ranks", (1, 1)),
        ("provider_ranks", ()),
        ("provider_ranks", [1]),
    ],
)
def test_host_consumer_rejects_invalid_route(field, value):
    with pytest.raises(ValueError):
        encode_domain_command(_command(replace(_grant(), **{field: value})))


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_instance_id", b"other123"),
        ("session_instance_id", bytes(8)),
        ("session_instance_id", b"short"),
        ("registry_epoch", -1),
        ("registry_epoch", 2**64),
        ("registry_epoch", True),
        ("endpoint_id", -1),
        ("endpoint_id", 2**32),
        ("endpoint_id", True),
    ],
)
def test_host_consumer_rejects_foreign_or_malformed_identity(field, value):
    grant = _grant()
    grant = replace(grant, consumer=replace(grant.consumer, **{field: value}))
    with pytest.raises(ValueError, match="identity"):
        encode_domain_command(_command(grant))


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", ce.AttachmentRole.PROVIDER),
        ("adapter_kind", None),
        ("adapter_profile", None),
        ("adapter_kind", ce.AdapterKind.DIRECT_MAP),
        ("adapter_profile", ce.AdapterProfile.HOST_VMM_COPY),
    ],
)
def test_host_consumer_rejects_nonconsumer_or_unimplemented_adapter(field, value):
    command = _command(_grant())
    row = list(command.attachments)
    row[1] = replace(row[1], **{field: value})
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, attachments=tuple(row)))


def test_host_consumer_rejects_duplicate_conflicting_or_missing_bindings():
    grant = _grant()
    second = replace(grant, sub_worker_id=2, consumer=replace(grant.consumer, endpoint_id=4))
    conflicts = [
        (grant, grant),
        (grant, replace(grant, provider_ranks=(0,))),
        (grant, replace(grant, sub_worker_id=2)),
        (grant, replace(second, consumer=replace(second.consumer, registry_epoch=1))),
    ]
    for grants in conflicts:
        with pytest.raises(ValueError):
            encode_domain_command(_command(*grants))
    with pytest.raises(ValueError, match="Host window"):
        encode_domain_command(replace(_command(grant), host_windows=()))
    with pytest.raises(ValueError):
        encode_domain_command(replace(_command(grant), attachments=()))
    with pytest.raises(ValueError, match="maximum"):
        encode_domain_command(_command(*([grant] * (GLOBAL_DOMAIN_MAX_ATTACHMENTS + 1))))


def test_compressed_grants_still_limit_expanded_rank_relations(monkeypatch):
    from simpler import global_comm_domain as gd  # noqa: PLC0415

    grant = replace(_grant(), provider_ranks=(0, 1))
    second = replace(grant, sub_worker_id=2, consumer=replace(grant.consumer, endpoint_id=4))
    encoded = encode_domain_command(_command(grant, second))
    monkeypatch.setattr(gd, "GLOBAL_DOMAIN_MAX_ATTACHMENTS", 3)
    for operation in (lambda: encode_domain_command(_command(grant, second)), lambda: decode_domain_command(encoded)):
        with pytest.raises(ValueError, match="maximum"):
            operation()


def test_host_consumer_rejects_phase_and_malformed_wire():
    encoded = encode_domain_command(_command(_grant()))
    for magic in (b"SDC1", b"SDC2", b"SDC9"):
        with pytest.raises(ValueError, match="format"):
            decode_domain_command(magic + encoded[4:])
    for size in range(1, 13):
        with pytest.raises(ValueError, match="truncated"):
            decode_domain_command(encoded[:-size])
    with pytest.raises(ValueError, match="trailing"):
        decode_domain_command(encoded + b"x")
    phase_changed = bytearray(encoded)
    struct.pack_into("<I", phase_changed, 4, int(GlobalDomainPhase.ABORT))
    with pytest.raises(ValueError, match="PREPARE_EXPORT"):
        decode_domain_command(bytes(phase_changed))
    # The optional table is the final field, following an otherwise complete command.
    count_offset = len(encode_domain_command(_command())) - 4
    oversized = bytearray(encoded)
    struct.pack_into("<I", oversized, count_offset, GLOBAL_DOMAIN_MAX_ATTACHMENTS + 1)
    with pytest.raises(ValueError, match="maximum"):
        decode_domain_command(bytes(oversized))
    # Grant header is node/sub, one endpoint identity, then the bounded rank list.
    identity_offset = count_offset + 4 + 8
    foreign = bytearray(encoded)
    foreign[identity_offset : identity_offset + 8] = b"other123"
    with pytest.raises(ValueError, match="identity"):
        decode_domain_command(bytes(foreign))
    for rank in (2, 0xFFFFFFFF):
        invalid = bytearray(encoded)
        struct.pack_into("<I", invalid, len(encoded) - 4, rank)
        with pytest.raises(ValueError):
            decode_domain_command(bytes(invalid))
    oversized_ranks = bytearray(encoded)
    struct.pack_into("<I", oversized_ranks, len(encoded) - 8, GLOBAL_DOMAIN_MAX_ATTACHMENTS + 1)
    with pytest.raises(ValueError, match="maximum"):
        decode_domain_command(bytes(oversized_ranks))
