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
from simpler.comm_domain import HostWindowSpec
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_MAX_ATTACHMENTS,
    GLOBAL_DOMAIN_VERSION,
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
        consumer_path="L4/L3[0]/SUB[1]",
        provider_rank=1,
        provider=ce.EndpointIdentity(b"creator1", 0, 7),
        attachment=ce.MemberAttachmentPlan(
            ce.EndpointIdentity(b"creator1", 0, 3),
            ce.AttachmentRole.CONSUMER,
            ce.AdapterKind.EXPLICIT_TRANSFER,
            ce.AdapterProfile.REMOTE_COPY,
        ),
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
    remote = _grant()
    local = replace(
        remote,
        provider_rank=0,
        provider=replace(remote.provider, endpoint_id=0),
        attachment=replace(
            remote.attachment,
            adapter_kind=ce.AdapterKind.OWNER_DELEGATED_COPY,
            adapter_profile=ce.AdapterProfile.HOST_VMM_COPY,
        ),
    )
    second_consumer = replace(
        remote,
        sub_worker_id=2,
        consumer_path="L4/L3[0]/SUB[2]",
        attachment=replace(remote.attachment, member=replace(remote.attachment.member, endpoint_id=4)),
    )
    command = _command(local, remote, second_consumer)
    encoded = encode_domain_command(command)
    assert encoded[:4] == b"SDC2"
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
    grants = root._global_host_consumer_attachments(command.members, (0, 1), "sim", command.window_size)
    cross_l3 = next(grant for grant in grants if grant.node_worker_id == 0 and grant.provider_rank == 1)
    registry = root._get_endpoint_registry()
    assert registry.same_node(cross_l3.provider, cross_l3.attachment.member)
    assert cross_l3.attachment.adapter_profile is ce.AdapterProfile.HOST_VMM_COPY
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
        ("provider_rank", -1),
        ("provider_rank", 2),
        ("provider_rank", True),
        ("consumer_path", "L3/SUB[1]"),
        ("consumer_path", "L4/L3[0]/SUB[0]"),
        ("consumer_path", "L4/L3[00]/SUB[1]"),
        ("consumer_path", "x" * 1025),
    ],
)
def test_host_consumer_rejects_invalid_route(field, value):
    with pytest.raises(ValueError):
        encode_domain_command(_command(replace(_grant(), **{field: value})))


@pytest.mark.parametrize("side", ["provider", "consumer"])
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
def test_host_consumer_rejects_foreign_or_malformed_identity(side, field, value):
    grant = _grant()
    if side == "provider":
        grant = replace(grant, provider=replace(grant.provider, **{field: value}))
    else:
        grant = replace(
            grant, attachment=replace(grant.attachment, member=replace(grant.attachment.member, **{field: value}))
        )
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
    grant = _grant()
    with pytest.raises(ValueError, match="adapter"):
        encode_domain_command(_command(replace(grant, attachment=replace(grant.attachment, **{field: value}))))


def test_host_consumer_rejects_duplicate_conflicting_or_missing_bindings():
    grant = _grant()
    conflicts = [
        (grant, grant),
        (replace(grant, provider=grant.attachment.member),),
        (
            replace(
                grant, attachment=replace(grant.attachment, member=replace(grant.attachment.member, registry_epoch=1))
            ),
        ),
        (grant, replace(grant, sub_worker_id=2, consumer_path="L4/L3[0]/SUB[2]")),
        (
            grant,
            replace(
                grant,
                sub_worker_id=2,
                consumer_path="L4/L3[0]/SUB[2]",
                attachment=replace(grant.attachment, member=replace(grant.attachment.member, endpoint_id=4)),
                provider=replace(grant.provider, endpoint_id=8),
            ),
        ),
    ]
    for grants in conflicts:
        with pytest.raises(ValueError):
            encode_domain_command(_command(*grants))
    with pytest.raises(ValueError, match="Host window"):
        encode_domain_command(replace(_command(grant), host_windows=()))
    with pytest.raises(ValueError, match="maximum"):
        encode_domain_command(_command(*([grant] * (GLOBAL_DOMAIN_MAX_ATTACHMENTS + 1))))


def test_host_consumer_rejects_phase_and_malformed_wire():
    encoded = encode_domain_command(_command(_grant()))
    for magic in (b"SDC1", b"SDC9"):
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
    for offset in (-12, -8, -4):
        bad_enum = bytearray(encoded)
        struct.pack_into("<I", bad_enum, len(encoded) + offset, 0xFFFFFFFF)
        with pytest.raises(ValueError):
            decode_domain_command(bytes(bad_enum))
    identity_offset = count_offset + 4 + 12 + 4 + len(_grant().consumer_path.encode())
    foreign = bytearray(encoded)
    foreign[identity_offset : identity_offset + 8] = b"other123"
    with pytest.raises(ValueError, match="identity"):
        decode_domain_command(bytes(foreign))
    bad_path_length = bytearray(encoded)
    struct.pack_into("<I", bad_path_length, count_offset + 4 + 12, 1025)
    with pytest.raises(ValueError, match="maximum"):
        decode_domain_command(bytes(bad_path_length))
