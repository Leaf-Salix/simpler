# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import os
import struct
import threading
from dataclasses import FrozenInstanceError, replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    buffer_reference,
)
from simpler.comm_domain import DomainConfig
from simpler.comm_endpoints import DEVICE_AICPU, HOST_CPU
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_BUFFER_REF_BYTES,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainBuffer,
    GlobalDomainCommand,
    GlobalDomainDescriptor,
    GlobalDomainHostWindow,
    GlobalDomainMember,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    GlobalDomainReleaseStage,
    decode_descriptor_table,
    decode_domain_command,
    decode_release_command,
    encode_descriptor_table,
    encode_domain_buffer_refs,
    encode_domain_command,
    encode_release_command,
)
from simpler.task_interface import HostWindowSpec
from simpler.worker import _host_domain_buffers


def _command():
    return GlobalDomainCommand(
        GlobalDomainPhase.PREPARE_EXPORT,
        7,
        3,
        "mixed",
        "sim",
        4096,
        (GlobalDomainMember(2, 0, 0, 0), GlobalDomainMember(5, 0, 1, 1)),
        (GlobalDomainBuffer("payload", 64),),
        owner_instance_id=b"creator1",
        host_windows=(
            GlobalDomainHostWindow(2, HostWindowSpec(0, 96), 12),
            GlobalDomainHostWindow(5, HostWindowSpec(1, 128), 13),
        ),
    )


def _exported_command():
    command = _command()
    return replace(
        command,
        phase=GlobalDomainPhase.IMPORT,
        descriptors=tuple(
            GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, 1, slot, 2, size, b"backing")
            for slot, size in enumerate((4096, 4096, 96, 128))
        ),
        backing_refs=tuple(
            BufferDescriptor(
                CanonicalIdentity(owner, 10, 1),
                AddressSpace.DEVICE,
                AccessMode.READWRITE,
                BackendKind.REMOTE_SIDECAR,
                64,
            )
            for owner in (b"hbmnode2", b"hbmnode5")
        ),
    )


def test_host_configuration_is_normalized_immutable_and_identity_free():
    request = {5: HostWindowSpec(1, 128), 2: HostWindowSpec(0, 96)}
    config = DomainConfig.capture(((2, 0), (5, 0)), 4096, (), True, (), request)
    assert config.host_windows == tuple(reversed(tuple(request.items())))
    assert config == DomainConfig.capture(config.members, 4096, (), True, (), dict(config.host_windows))
    request[2] = HostWindowSpec(0, 64)
    assert config.host_windows[0][1].nbytes == 96
    with pytest.raises(FrozenInstanceError):
        config.host_windows[0][1].nbytes = 8
    changed = DomainConfig.capture(config.members, 4096, (), True, (), request)
    assert config.difference(changed).startswith("host_windows:")
    with pytest.raises(ValueError, match="exactly"):
        DomainConfig.capture(config.members, 4096, (), True, (), {2: HostWindowSpec(0, 96)})
    with pytest.raises(ValueError, match="L4"):
        DomainConfig.capture((0, 1), 4096, (), True, (), request)


@pytest.mark.parametrize(("numa", "size"), [(-1, 32), (0, 0), (0, -1), (2**31, 32), (0, 2**64)])
def test_host_spec_rejects_invalid_location_and_extent(numa, size):
    with pytest.raises(ValueError):
        HostWindowSpec(numa, size)


@pytest.mark.parametrize(
    "phase", [GlobalDomainPhase.PREPARE_EXPORT, GlobalDomainPhase.IMPORT, GlobalDomainPhase.COMMIT]
)
def test_host_command_and_logical_buffers_preserve_identity(phase):
    command = _command()
    if phase != GlobalDomainPhase.PREPARE_EXPORT:
        command = replace(_exported_command(), phase=phase)
    received = decode_domain_command(encode_domain_command(command))
    assert received == command
    logical = _host_domain_buffers(command)[5]
    mapped = replace(_host_domain_buffers(received)[5], base=8192)
    assert logical.to_descriptor() == mapped.to_descriptor() and mapped.base == 8192
    assert logical.backend_kind == BackendKind.REMOTE_SIDECAR and not logical.body
    assert logical.identity != _exported_command().backing_refs[1].identity
    with pytest.raises(KeyError):
        _host_domain_buffers(command)[99]
    exports = _exported_command().descriptors
    assert decode_descriptor_table(encode_descriptor_table(exports)) == exports
    with pytest.raises(ValueError, match="trailing"):
        decode_descriptor_table(encode_descriptor_table(exports) + b"x")


@pytest.mark.parametrize("mutation", ["order", "collision", "missing", "unprepared", "creator"])
def test_host_wire_rejects_conflicting_or_incomplete_metadata(mutation):
    command = _command()
    if mutation == "order":
        command = replace(command, host_windows=command.host_windows[::-1])
    elif mutation == "collision":
        command = replace(
            command, host_windows=(replace(command.host_windows[0], buffer_id=13), command.host_windows[1])
        )
    elif mutation == "missing":
        command = replace(command, host_windows=command.host_windows[:1])
    elif mutation == "unprepared":
        command = replace(command, descriptors=_exported_command().descriptors)
    else:
        command = replace(command, owner_instance_id=b"")
    with pytest.raises(ValueError):
        encode_domain_command(command)


def test_release_stage_and_old_wire_are_strict():
    for stage in GlobalDomainReleaseStage:
        command = GlobalDomainReleaseCommand(7, 3, stage)
        assert decode_release_command(encode_release_command(command)) == command
    with pytest.raises(ValueError, match="size"):
        decode_release_command(struct.pack("<QQ", 7, 3))
    with pytest.raises(ValueError):
        decode_release_command(struct.pack("<QQI", 7, 3, 99))
    encoded = encode_domain_command(_command())
    with pytest.raises(ValueError, match="format"):
        decode_domain_command(b"SDC9" + encoded[4:])
    with pytest.raises(ValueError, match="trailing"):
        decode_domain_command(encoded + b"x")


@pytest.mark.parametrize("tag", [7, 128, 255])
def test_native_descriptor_rejects_retired_and_unknown_domain_backends(tag):
    from _task_interface import _decode_buffer_descriptor_wire, _encode_buffer_descriptor_wire  # noqa: PLC0415

    wire = bytearray(_encode_buffer_descriptor_wire(_exported_command().backing_refs[0]))
    wire[4] = tag
    with pytest.raises(ValueError):
        _decode_buffer_descriptor_wire(bytes(wire))


@pytest.mark.parametrize(
    "slot,field,value",
    [
        (3, "domain_rank", 2),
        (3, "domain_rank", 4),
        (3, "mapping_size", 127),
        (1, "mapping_size", 8192),
        (3, "rank_count", 3),
        (3, "version", 2),
        (3, "profile_id", 2),
        (3, "handle", b""),
        (3, "handle", b"x" * 257),
        (3, "mapping_size", 0),
        (3, "mapping_size", 2**64),
    ],
)
def test_window_exports_reject_invalid_metadata(slot, field, value):
    command = _exported_command()
    descriptors = list(command.descriptors)
    descriptors[slot] = replace(descriptors[slot], **{field: value})
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, descriptors=tuple(descriptors)))


@pytest.mark.parametrize("field,value", [("node", -1), ("identity", 0)])
def test_host_config_rejects_invalid_wire(field, value):
    wire = bytearray(encode_domain_command(_exported_command()))
    first_host = len(wire) - 4 - 2 * GLOBAL_DOMAIN_BUFFER_REF_BYTES - 2 * struct.calcsize("<iiQQ")
    struct.pack_into("<i" if field == "node" else "<Q", wire, first_host + (0 if field == "node" else 16), value)
    with pytest.raises(ValueError):
        decode_domain_command(bytes(wire))


def test_window_exports_reject_missing_and_excess_entries():
    command = _exported_command()
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, descriptors=command.descriptors[:-1]))
    with pytest.raises(ValueError):
        encode_descriptor_table(command.descriptors * 33)


@pytest.mark.parametrize("mutation", ["wrong-node", "duplicate-slot", "missing-host"])
def test_exports_collector_checks_node_ownership(monkeypatch, mutation):
    from simpler.worker import Worker  # noqa: PLC0415

    command = _exported_command()
    exports = command.descriptors
    replies: dict[int, tuple[GlobalDomainDescriptor, ...]] = {2: (exports[0], exports[2]), 5: (exports[1], exports[3])}
    if mutation == "wrong-node":
        replies[2], replies[5] = replies[5], replies[2]
    elif mutation == "duplicate-slot":
        replies[2] = (exports[0], exports[0])
    else:
        replies[5] = replies[5][:1]
    worker = Worker(level=4, num_sub_workers=0)
    monkeypatch.setattr(
        worker,
        "_global_domain_control",
        lambda node, *_args: encode_descriptor_table(replies[node])
        + encode_domain_buffer_refs(((command.backing_refs[(2, 5).index(node)], 0, 0),)),
    )
    try:
        with pytest.raises(RuntimeError, match="node windows"):
            worker._collect_global_domain_exports(
                replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backing_refs=()), (2, 5), []
            )
    finally:
        worker.close()


@pytest.mark.parametrize("length", [0, 127, 129])
def test_host_fabric_export_requires_exact_token_size(length):
    command = _exported_command()
    exports = tuple(
        replace(d, profile_id=2, handle=b"x" * (length if d.domain_rank == 3 else 128)) for d in command.descriptors
    )
    with pytest.raises(ValueError):
        encode_domain_command(replace(command, profile="a3-fabric-v1", descriptors=exports))


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize("oversized", [False, True])
def test_native_host_views_copy_and_two_stage_release(arch, runtime, oversized):
    root = Path(__file__).resolve().parents[3] / "build/lib"
    ctypes.CDLL(str(root / "libcpu_sim_context.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(root / arch / "sim" / runtime / "libhost_runtime.so"))
    u64, u32, size, ptr = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p
    signatures = {
        "prepare": [u64, u32, u32, size, u32, u32, ptr, ptr],
        "import": [u64, ptr, size, ptr],
        "window": [u64, u32, ptr, ptr],
        "copy": [u64, u32, size, u32, size, size, ptr],
        "release": [u64, u32],
    }
    for name, args in signatures.items():
        function = getattr(lib, "comm_global_domain_" + name)
        function.argtypes, function.restype = args, ctypes.c_int
    domain_id = int.from_bytes(os.urandom(8), "little") or 1
    owner = SharedMemory(create=True, size=128)
    try:
        assert owner.buf is not None
        owner.buf[:128] = bytes(range(128))
        descriptor, local_base = ctypes.create_string_buffer(288), u64()
        assert lib.comm_global_domain_prepare(domain_id, 0, 1, 128, 1, 1, descriptor, ctypes.byref(local_base)) == 0
        token = ("/" + owner.name).encode()
        host = struct.pack(
            "<IIIIQII256s",
            GLOBAL_DOMAIN_VERSION,
            1,
            1,
            1,
            owner.size + 4096 if oversized else 128,
            len(token),
            0,
            token,
        )
        context = u64()
        table = ctypes.create_string_buffer(descriptor.raw + host)
        rc = getattr(lib, "comm_global_domain_import")(domain_id, table, 2, ctypes.byref(context))
        if oversized:
            assert rc != 0
            return
        assert rc == 0
        mapped, mapped_size = u64(), size()
        assert lib.comm_global_domain_window(domain_id, 1, ctypes.byref(mapped), ctypes.byref(mapped_size)) == 0
        assert mapped_size.value == 128
        assert ctypes.string_at(mapped.value, 128) == bytes(range(128))
        assert lib.comm_global_domain_copy(domain_id, 0, 32, 1, 16, 32, None) == 0
        assert lib.comm_global_domain_copy(domain_id, 1, 64, 0, 32, 32, None) == 0
        expected = bytearray(range(128))
        expected[64:96] = bytes(range(16, 48))
        assert bytes(owner.buf[:128]) == expected  # Includes both unchanged guards.
        assert lib.comm_global_domain_copy(domain_id, 1, 16, 1, 20, 32, None) != 0
        assert lib.comm_global_domain_copy(domain_id, 1, 120, 0, 0, 16, None) != 0
        assert lib.comm_global_domain_release(domain_id, 0) != 0
        assert lib.comm_global_domain_release(domain_id, 2) != 0
        assert lib.comm_global_domain_release(domain_id, 1) == 0
        assert lib.comm_global_domain_window(domain_id, 1, ctypes.byref(mapped), ctypes.byref(mapped_size)) != 0
        assert lib.comm_global_domain_copy(domain_id, 1, 64, 0, 32, 32, None) != 0
        assert ctypes.string_at(local_base.value, 16) == bytes(16)  # HBM owner still alive after VIEWS.
        assert lib.comm_global_domain_release(domain_id, 2) == 0
        assert lib.comm_global_domain_release(domain_id, 2) == 0
    finally:
        lib.comm_global_domain_release(domain_id, 1)
        lib.comm_global_domain_release(domain_id, 2)
        owner.close()
        owner.unlink()


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_native_partial_import_keeps_views_unpublished_until_release(arch, runtime):
    root = Path(__file__).resolve().parents[3] / "build/lib"
    ctypes.CDLL(str(root / "libcpu_sim_context.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(root / arch / "sim" / runtime / "libhost_runtime.so"))
    u64, u32, size, ptr = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p
    for name, args in {
        "prepare": [u64, u32, u32, size, u32, u32, ptr, ptr],
        "import": [u64, ptr, size, ptr],
        "window": [u64, u32, ptr, ptr],
        "copy": [u64, u32, size, u32, size, size, ptr],
        "release": [u64, u32],
    }.items():
        function = getattr(lib, "comm_global_domain_" + name)
        function.argtypes, function.restype = args, ctypes.c_int
    domain_ids = tuple(int.from_bytes(os.urandom(8), "little") or 1 for _ in range(2))
    owner = SharedMemory(create=True, size=128)
    try:
        assert owner.buf is not None
        expected = bytes(range(128))
        owner.buf[:128] = expected
        descriptors, local_bases = [], []
        for rank, domain_id in enumerate(domain_ids):
            descriptor, base = ctypes.create_string_buffer(288), u64()
            assert lib.comm_global_domain_prepare(domain_id, rank, 2, 128, 1, 1, descriptor, ctypes.byref(base)) == 0
            descriptors.append(descriptor.raw)
            local_bases.append(base.value)
        host = GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, 1, 2, 2, 128, ("/" + owner.name).encode())
        missing = replace(host, domain_rank=3, handle=("/missing_" + os.urandom(8).hex()).encode())
        table = ctypes.create_string_buffer(b"".join(descriptors) + host.encode() + missing.encode())
        context = u64()
        assert getattr(lib, "comm_global_domain_import")(domain_ids[0], table, 4, ctypes.byref(context)) != 0
        assert context.value == 0
        assert getattr(lib, "comm_global_domain_import")(domain_ids[0], table, 4, ctypes.byref(context)) != 0
        for slot in range(4):
            mapped, mapped_size = u64(), size()
            assert (
                lib.comm_global_domain_window(domain_ids[0], slot, ctypes.byref(mapped), ctypes.byref(mapped_size)) != 0
            )
        assert lib.comm_global_domain_copy(domain_ids[0], 2, 32, 2, 0, 16, None) != 0
        assert lib.comm_global_domain_copy(domain_ids[0], 0, 32, 2, 0, 16, None) != 0
        assert bytes(owner.buf[:128]) == expected
        assert lib.comm_global_domain_release(domain_ids[0], 2) != 0
        assert lib.comm_global_domain_release(domain_ids[0], 1) == 0
        assert ctypes.string_at(local_bases[0], 128) == bytes(128)
        assert bytes(owner.buf[:128]) == expected
        assert lib.comm_global_domain_release(domain_ids[0], 2) == 0
    finally:
        for domain_id in domain_ids:
            lib.comm_global_domain_release(domain_id, 1)
            lib.comm_global_domain_release(domain_id, 2)
        owner.close()
        owner.unlink()


def test_two_node_host_cpu_copy_reuse_and_stale_access():
    from simpler.comm_domain import CommDomainRef  # noqa: PLC0415
    from simpler.task_interface import CommBufferSpec, DataType, TaskArgs  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    saved_parts = []
    nodes = (0, 1)

    def node_task(orch, args, _config):
        node, phase = int(args.scalar(0)), int(args.scalar(1))
        if phase == 2:
            with pytest.raises((KeyError, RuntimeError)):
                saved_parts[0].read(0, bytearray(1))
            return
        domain = orch.get_domain(
            CommDomainRef(int(args.scalar(2)).to_bytes(8, "little"), int(args.scalar(3)), int(args.scalar(4)))
        )
        part = domain.local_host()
        local = domain.host(node)
        if phase == 0:
            saved_parts.append(part)
            data = bytes([node + 11]) * 128 + bytes([0xCC]) * (1536 - 128)
            part.write(0, data)
            for member in domain.members:
                if member.node_worker_id == node:
                    orch._worker.copy_domain(
                        member.buffers["payload"], local, worker=member.local_worker_id, nbytes=512
                    )
            return
        pools = [(domain.host(n), n) for n in nodes]
        pools += [(member.buffers["payload"], member.node_worker_id) for member in domain.members]
        for chip in (0, 1):
            for pool, owner in pools:
                orch.copy_domain(local, pool, worker=chip, nbytes=128, dst_offset=512)
                result = bytearray(128)
                part.read(512, result)
                assert result == bytes([owner + 11]) * 128
                orch.copy_domain(pool, local, worker=chip, nbytes=64, dst_offset=128)
                orch.copy_domain(local, pool, worker=chip, nbytes=512, dst_offset=512)
                result = bytearray(512)
                part.read(512, result)
                assert result == bytes([owner + 11]) * 128 + bytes([node + 11]) * 64 + bytes([0xCC]) * 320
        with pytest.raises(ValueError, match="overlap"):
            orch.copy_domain(local, local, worker=0, nbytes=64, dst_offset=16)
        with pytest.raises(ValueError, match="range"):
            orch.copy_domain(local, local, worker=0, nbytes=64, dst_offset=1530)
        with pytest.raises(ValueError, match="granted"):
            orch.copy_domain(replace(local, nbytes=1024), local, worker=0, nbytes=64)

    root = Worker(level=4, num_sub_workers=0)
    task = root.register(node_task)
    captured = []
    try:
        for node in nodes:
            assert (
                root.add_worker(
                    Worker(
                        level=3,
                        platform="a2a3sim",
                        runtime="tensormap_and_ringbuffer",
                        device_ids=[0, 1],
                        num_sub_workers=0,
                        comm_profile="sim",
                        global_device_ranks=[2 * node, 2 * node + 1],
                    )
                )
                == node
            )
        root.init()
        request = dict(
            name="mixed",
            members=tuple((n, c) for n in nodes for c in (0, 1)),
            window_size=512,
            buffers=[CommBufferSpec("payload", "uint8", 512, 512)],
            retain_after_run=True,
            host_windows={n: HostWindowSpec(0, 1536) for n in nodes},
        )

        def allocate(orch, _args, _config):
            domain = orch.get_or_allocate_domain(**request)
            assert orch.get_or_allocate_domain(**request) is domain
            captured.append(domain)
            with pytest.raises(RuntimeError, match="no local Host"):
                domain.local_host()

        root.run(allocate)
        domain = captured[0]
        assert len({b.identity for b in domain._host_buffers.values()}) == 2
        for phase in (0, 1, 2):
            if phase == 2:
                domain.release()
                assert domain.freed
            for node in nodes:

                def submit(orch, _args, _config, node=node, phase=phase):
                    args = TaskArgs()
                    if phase != 2:
                        current = orch.get_domain(domain.ref)
                        args.add_tensor(current.host(node).tensor([1536], DataType.UINT8))
                    args.add_scalar(node)
                    args.add_scalar(phase)
                    if phase != 2:
                        args.add_scalar(int.from_bytes(current.ref.owner_instance_id, "little"))
                        args.add_scalar(current.ref.domain_id)
                        args.add_scalar(current.ref.generation)
                    orch.submit_next_level(task, args, worker=node)

                root.run(submit)
        assert not root.live_domains
    finally:
        root.close()


def _control_payload(data):
    shm = SharedMemory(create=True, size=len(data))
    shm.unlink()
    assert shm.buf is not None
    shm.buf[: len(data)] = data
    return shm, shm.buf[: len(data)], len(data)


@pytest.mark.parametrize("failure", [None, (1, 2), (1, 5), (2, 2), (2, 5)])
def test_owner_release_requires_every_node_view_ack(monkeypatch, failure):
    from simpler.remote_l3_protocol import ControlName  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker, calls = Worker(level=4, num_sub_workers=0), []

    def control(node, operation, payload):
        assert operation == ControlName.RELEASE_DOMAIN
        command = decode_release_command(payload)
        calls.append((int(command.stage), node))
        if calls[-1] == failure:
            raise RuntimeError("lost ACK")
        return b""

    monkeypatch.setattr(worker, "_global_domain_control", control)
    monkeypatch.setattr(
        worker, "_mpi_group_for_involved_nodes", lambda _nodes: pytest.fail("Host release must use per-node control")
    )
    try:
        if failure:
            with pytest.raises(RuntimeError, match="unconfirmed"):
                worker._release_host_domain_nodes(7, 3, (2, 5))
        else:
            worker._release_host_domain_nodes(7, 3, (2, 5))
        assert calls == ([(1, 2), (1, 5)] if failure and failure[0] == 1 else [(1, 2), (1, 5), (2, 2), (2, 5)])
    finally:
        worker.close()


def test_native_prepare_failure_remains_reachable_for_staged_release(monkeypatch):
    import simpler.worker as module  # noqa: PLC0415

    store, calls = module._L2GlobalDomainStore(), []

    class Chip:
        def comm_global_domain_prepare(self, *_args):
            raise RuntimeError("physical allocation reply lost")

        def comm_global_domain_release(self, identity, stage):
            calls.append((identity, stage))

    data = bytearray(module.LOCAL_PREPARE_REPLY.size + module.GLOBAL_DOMAIN_DESCRIPTOR_BYTES)
    module.LOCAL_PREPARE_REQUEST.pack_into(
        data, 0, module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 0, 1, 1, 128, True
    )
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(data))
    with pytest.raises(RuntimeError, match="reply lost"):
        module._handle_ctrl_global_domain_prepare(Chip(), memoryview(bytearray()), store)
    assert store.domains[7].descriptor is None and store.domains[7].host_enabled
    for stage in (1, 2):
        data = module.LOCAL_RELEASE_REQUEST.pack(module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, stage)
        module._handle_ctrl_global_domain_release(Chip(), memoryview(bytearray()), store)
        assert (7 in store.domains) == (stage == 1)
    assert calls == [(7, 1), (7, 2)]


@pytest.mark.parametrize("mode", ["node", "cached-root", "untracked-root"])
def test_host_close_cannot_bypass_global_release(monkeypatch, mode):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.task_interface import CommDomainHandle  # noqa: PLC0415

    worker, shutdown = module.Worker(level=3 if mode == "node" else 4, num_sub_workers=0), []
    state = module._GlobalNodeDomainState(_command())
    owner = object()
    state.host_owner = owner
    if mode == "node":
        worker._global_node_domains[7] = state
    elif mode == "cached-root":
        handle = CommDomainHandle(
            name="mixed",
            _release_fn=lambda _handle: None,
            domain_id=7,
            generation=3,
            members=_command().members,
            host_buffers={2: _host_domain_buffers(_command())[2]},
        )
        worker._live_global_domains["mixed"] = handle
        worker._global_domain_free_results[7] = RuntimeError("lost view ACK")
        worker._defer_domain_retirement(handle)
    else:
        worker._record_unreclaimable("Host rollback unknown", keep_transport=True)
    monkeypatch.setattr(worker, "_release_remote_sessions", lambda: shutdown.append(True))
    try:
        with pytest.raises(RuntimeError):
            worker.close()
        assert state.host_owner is owner and not shutdown
        if mode == "node":
            assert worker._global_node_domains[7] is state
        elif mode == "cached-root":
            assert not worker._cleanup_journal.empty and not handle.freed
            assert worker._global_domain_free_results[7] is not None
    finally:
        # Only synthetic debts: no backing, native context or child process was created.
        worker._global_node_domains.clear()
        worker._live_global_domains.clear()
        worker._global_domain_free_results.clear()
        worker._cleanup_journal = type(worker._cleanup_journal)()
        with pytest.raises(RuntimeError):  # A failed close preserves its published error.
            worker.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "unknown",
        "stale",
        "overlap",
        "range",
        "read-only",
        "revoked",
        "generation",
        "truncated",
        "native-failure",
    ],
)
def test_pool_copy_reuses_grants_and_revokes_unknown_completion(monkeypatch, mutation):
    import simpler.worker as module  # noqa: PLC0415

    command = _exported_command()
    host, peer = _host_domain_buffers(command)[2], buffer_reference(command.backing_refs[1])
    slots = module._domain_buffer_slots(command)
    entry = module._L2GlobalDomain(
        7,
        3,
        0,
        2,
        None,
        0,
        4096,
        4096,
        device_ctx=123,
        host_enabled=True,
        grants={identity: slots[identity] for identity in (host.identity, peer.identity)},
    )
    store = module._L2GlobalDomainStore({7: entry})
    dst, src, offset = host, peer, 8
    if mutation in ("unknown", "stale"):
        identity = host.identity
        dst = replace(
            host,
            identity=CanonicalIdentity(
                identity.owner_instance_id,
                identity.buffer_id + (mutation == "unknown"),
                identity.generation + (mutation == "stale"),
            ),
        )
    elif mutation == "overlap":
        src, offset = host, 2
    elif mutation == "range":
        offset = host.nbytes
    elif mutation == "read-only":
        entry.grants[host.identity] = (replace(host, access=AccessMode.READ).to_descriptor(), *slots[host.identity][1:])
    elif mutation == "revoked":
        entry.revoked = True
    elif mutation == "generation":
        entry.generation += 1
    payload = module.LOCAL_POOL_COPY.pack(module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, offset, 4, 8)
    payload += module._pack_identity_wire(dst.identity) + module._pack_identity_wire(src.identity)
    if mutation == "truncated":
        payload = payload[:-1]
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(payload))
    calls = []

    class Chip:
        def comm_global_domain_copy(self, *args):
            calls.append(args)
            if mutation == "native-failure":
                raise RuntimeError("completion unknown")

    if mutation == "valid":
        module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store)
        assert calls == [(7, 2, 8, 1, 4, 8)]
    elif mutation == "native-failure":
        with pytest.raises(RuntimeError, match="unknown"):
            module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store)
        assert entry.revoked
        with pytest.raises(ValueError, match="live Host domain grant"):
            module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store)
    else:
        with pytest.raises(ValueError):
            module._handle_ctrl_domain_pool_copy(Chip(), memoryview(bytearray()), store)
        assert not calls and entry.revoked == (mutation == "revoked")


@pytest.mark.parametrize("field,value", [("handle", b"changed"), ("mapping_size", 8192)])
def test_import_freezes_this_nodes_exports_before_dispatch(field, value):
    import simpler.worker as module  # noqa: PLC0415

    command = _exported_command()
    worker = module.Worker(level=3, num_sub_workers=0)
    state = module._GlobalNodeDomainState(
        replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backing_refs=()), node_worker_id=2
    )
    state.descriptors = {0: command.descriptors[0], 2: command.descriptors[2]}
    worker._global_node_domains[7] = state
    worker._worker = object()
    changed = list(command.descriptors)
    changed[2] = replace(changed[2], **{field: value})
    try:
        with pytest.raises(RuntimeError, match="exported backing"):
            worker._import_global_domain_node(replace(command, descriptors=tuple(changed)), 2)
    finally:
        worker._worker = None
        worker._global_node_domains.clear()
        worker.close()


@pytest.mark.parametrize("field,value", [("handle", b"changed"), ("mapping_size", 8192)])
def test_commit_freezes_complete_host_exports(field, value):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainView  # noqa: PLC0415

    command = _exported_command()
    worker = module.Worker(level=3, num_sub_workers=0)
    view = GlobalCommDomainView(
        name="mixed", members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
    )
    state = module._GlobalNodeDomainState(command, phase=GlobalDomainPhase.IMPORT, view=view)
    worker._global_node_domains[7] = state
    try:
        commit = replace(command, phase=GlobalDomainPhase.COMMIT)
        changed = replace(
            commit, descriptors=(*commit.descriptors[:-1], replace(commit.descriptors[-1], **{field: value}))
        )
        with pytest.raises(RuntimeError, match="conflicts"):
            worker._commit_global_domain_node(changed)
        assert not view.committed
        worker._commit_global_domain_node(commit)
        assert view.committed
    finally:
        worker._global_node_domains.clear()
        worker.close()


def test_shared_host_targets_preserve_exact_descriptor_and_hbm_chip_checks():
    from simpler.task_interface import DataType, TaskArgs  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    command = _exported_command()
    host = _host_domain_buffers(command)[2]
    hbm = buffer_reference(command.backing_refs[0], owner_worker_id=2)
    try:
        with worker._child_prov_lock:
            worker._record_device_alloc(host)
            for target in (2, 5):
                worker._domain_buffer_routes[(host.identity, target)] = buffer_reference(host, owner_worker_id=target)
            worker._record_device_alloc(hbm)
            args = TaskArgs()
            args.add_tensor(host.tensor([8], DataType.UINT8))
            for target in (2, 5):
                worker._prepare_buffer_references(args, None, target)
                worker._child_prov_check_dispatch_locked(args, target, api="test")
            with pytest.raises(ValueError, match="another worker|stale"):
                worker._prepare_buffer_references(args, None, 999)
            changed = TaskArgs()
            changed.add_tensor(replace(host, nbytes=64).tensor([8], DataType.UINT8))
            with pytest.raises(ValueError, match="differs from its registration"):
                worker._prepare_buffer_references(changed, None, 2)
            args.add_tensor(hbm.tensor([8], DataType.UINT8))
            with pytest.raises(ValueError, match="another worker|stale"):
                worker._prepare_buffer_references(args, None, 5)
        with pytest.raises(ValueError, match="explicit copy_domain"):
            worker._device_worker_for(host.identity, api="test")
    finally:
        worker._clear_child_prov()
        worker.close()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "host-collision", "extent", "access", "space", "path", "backend", "phase", "no-host"],
)
def test_command_backing_references_reject_conflicting_grants(mutation):
    command = _exported_command()
    first, second = command.backing_refs
    changes = {
        "host-collision": {
            "identity": CanonicalIdentity(command.owner_instance_id, command.host_windows[0].buffer_id, 1)
        },
        "extent": {"nbytes": 128},
        "access": {"access": AccessMode.READ},
        "space": {"address_space": AddressSpace.HOST},
        "path": {"owner_worker_path_id": 1},
        "backend": {"backend_kind": BackendKind.VMM_WINDOW, "body": (4096).to_bytes(8, "little")},
    }
    if mutation in changes:
        command = replace(
            command, backing_refs=(replace(buffer_reference(first), **changes[mutation]).to_descriptor(), second)
        )
    elif mutation == "missing":
        command = replace(command, backing_refs=(first,))
    elif mutation == "duplicate":
        command = replace(command, backing_refs=(first, first))
    elif mutation == "phase":
        command = replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=())
    else:
        command = replace(command, host_windows=(), descriptors=command.descriptors[:2])
    with pytest.raises(ValueError, match="backing reference"):
        encode_domain_command(command)


@pytest.mark.parametrize("mutation", ["locator", "count", "truncated"])
def test_command_backing_reference_wire_rejects_malformed_payload(mutation):
    wire = bytearray(encode_domain_command(_exported_command()))
    count_at = len(wire) - 2 * GLOBAL_DOMAIN_BUFFER_REF_BYTES - 4
    if mutation == "locator":
        struct.pack_into("<QQ", wire, count_at + 4 + 88, 1, 1)
    elif mutation == "count":
        struct.pack_into("<I", wire, count_at, 4097)
    else:
        del wire[-1]
    with pytest.raises(ValueError):
        decode_domain_command(bytes(wire))


def test_borrowed_host_identity_has_endpoint_local_mappings_and_revocation():
    descriptor = _host_domain_buffers(_command())[2].to_descriptor()
    registries = [ImportRegistry(ImportContext(DEVICE_AICPU)) for _ in range(2)]
    host = ImportRegistry(ImportContext(HOST_CPU))
    try:
        for registry, base in zip(registries, (4096, 8192)):
            registry.register_borrowed(descriptor, base)
            registry.register_borrowed(descriptor, base)
            assert registry.materialize(descriptor).base == base
            with pytest.raises(ValueError, match="conflict"):
                registry.register_borrowed(descriptor, base + 64)
            changed = replace(buffer_reference(descriptor), nbytes=64).to_descriptor()
            with pytest.raises(ValueError, match="conflict"):
                registry.register_borrowed(changed, base)
            with pytest.raises(ValueError, match="different descriptor"):
                registry.materialize(changed)
        with pytest.raises(ValueError, match="granted device"):
            host.register_borrowed(descriptor, 4096)
        registries[0].unregister(descriptor.identity)
        with pytest.raises(KeyError, match="no buffer registered"):
            registries[0].require(descriptor.identity)
        assert registries[1].require(descriptor.identity).base == 8192
        with pytest.raises(ValueError):
            registries[0].materialize(descriptor)
    finally:
        for registry in (*registries, host):
            registry.close()


def test_shared_host_reference_uses_each_receiving_session_locator():
    from simpler.task_interface import DataType, TaskArgs, _RemoteTaskArgsSidecar  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=4, num_sub_workers=0)
    public = _host_domain_buffers(_command())[2]
    args = TaskArgs()
    args.add_tensor(public.tensor([8], DataType.UINT8, byte_offset=4))
    sidecar = _RemoteTaskArgsSidecar((None,), b"")
    try:
        with worker._child_prov_lock:
            worker._record_device_alloc(public)
            for target, locator in ((2, (41, 3)), (5, (62, 4))):
                worker._domain_buffer_routes[(public.identity, target)] = buffer_reference(
                    public, owner_worker_id=target, remote_ref=locator
                )
                prepared = worker._prepare_buffer_references(args, sidecar, target)
                assert prepared is not None and prepared.tensors[0] is not None
                descriptor = prepared.tensors[0].desc
                assert (descriptor.owner_worker_id, descriptor.buffer_id, descriptor.generation) == (target, *locator)
                assert descriptor.offset == 4 and descriptor.nbytes == public.nbytes - 4
                worker._child_prov_check_dispatch_locked(args, target, api="test")
            assert public._remote_ref is None and sidecar.tensors == (None,)
            with pytest.raises(ValueError, match="another worker|stale"):
                worker._prepare_buffer_references(args, sidecar, 99)
            worker._drop_device_alloc(public.identity)
            assert not worker._domain_buffer_routes
            with pytest.raises(ValueError, match="stale"):
                worker._prepare_buffer_references(args, sidecar, 2)
    finally:
        worker._clear_child_prov()
        worker.close()


@pytest.mark.parametrize("mutation", [None, "missing", "duplicate", "locator"])
def test_exports_collector_preserves_rank_order_and_rejects_conflicting_references(monkeypatch, mutation):
    from simpler.worker import Worker  # noqa: PLC0415

    command = _exported_command()
    worker = Worker(level=4, num_sub_workers=0)
    replies = {}
    for index, node in enumerate((2, 5)):
        refs = [(command.backing_refs[index], int(mutation == "locator"), 0)]
        if mutation == "missing" and node == 5:
            refs = []
        elif mutation == "duplicate" and node == 5:
            refs = [(command.backing_refs[0], 0, 0)]
        replies[node] = encode_descriptor_table((command.descriptors[index], command.descriptors[index + 2]))
        replies[node] += encode_domain_buffer_refs(refs)
    monkeypatch.setattr(worker, "_global_domain_control", lambda node, *_args: replies[node])
    prepared = []
    try:
        base = replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backing_refs=())
        if mutation:
            with pytest.raises(ValueError, match="backing references"):
                worker._collect_global_domain_exports(base, (2, 5), prepared)
        else:
            assert worker._collect_global_domain_exports(base, (2, 5), prepared) == (
                command.descriptors,
                command.backing_refs,
            )
            assert prepared == [2, 5]
    finally:
        worker.close()


@pytest.mark.parametrize("profile", ["sim", "a3-fabric-v1"])
def test_borrowed_host_access_outside_callback_does_not_reenter_serializer(monkeypatch, profile):
    import simpler.worker as module  # noqa: PLC0415
    from simpler.comm_domain import CommDomainRef  # noqa: PLC0415
    from simpler.task_interface import GlobalCommDomainView  # noqa: PLC0415

    command = replace(_exported_command(), profile=profile)
    worker = module.Worker(level=3, num_sub_workers=0)
    owner = SharedMemory(create=True, size=96)
    assert owner.buf is not None
    owner.buf[:] = bytes(len(owner.buf))
    base = ctypes.addressof(ctypes.c_char.from_buffer(owner.buf))
    calls = []

    class HostOwner:
        def write_bytes(self, offset, pointer, nbytes):
            calls.append(("write", offset, pointer, nbytes))
            ctypes.memmove(base + offset, pointer, nbytes)

        def read_bytes(self, offset, pointer, nbytes):
            calls.append(("read", offset, pointer, nbytes))
            ctypes.memmove(pointer, base + offset, nbytes)

    view = GlobalCommDomainView(
        name=command.name, members=command.members, contexts={}, domain_id=7, generation=3, mapping_size=4096
    )
    state = module._GlobalNodeDomainState(
        command,
        phase=GlobalDomainPhase.COMMIT,
        view=view,
        host_owner=owner if profile == "sim" else HostOwner(),
        node_worker_id=2,
        host_buffers=_host_domain_buffers(command),
    )
    worker._global_node_domains[7] = state
    worker._lifecycle = module._Lifecycle.READY
    errors = []
    completed = []
    memmove = ctypes.memmove

    def direct_copy(dst, src, size):
        assert isinstance(dst, int) and isinstance(src, int)
        return memmove(dst, src, size)

    def access():
        try:
            assert module._callback_frame_for(worker) is None
            ref = CommDomainRef(command.owner_instance_id, 7, 3)
            handle = worker._get_domain(ref)
            assert handle.borrowed and worker._get_domain(ref) is handle
            part = handle.local_host()
            source = bytearray(b"head" + bytes(range(96)) + b"tail")
            result = bytearray([0xCC] * 80)
            with monkeypatch.context() as copy_patch:
                copy_patch.setattr(module.ctypes, "memmove", direct_copy)
                copy_patch.setattr(module.ctypes, "string_at", lambda *_: pytest.fail("unexpected bytes staging"))
                part.write(8, memoryview(source)[4:-4], 64)
                part.read(8, memoryview(result)[8:72])
                with pytest.raises(ValueError, match="exceeds part"):
                    part.write(80, source, 32)
                with pytest.raises(ValueError, match="writable"):
                    part.read(0, bytes(4))
            assert result == bytes([0xCC] * 8) + bytes(range(64)) + bytes([0xCC] * 8)
            assert source == b"head" + bytes(range(96)) + b"tail"
            assert owner.buf is not None
            assert bytes(owner.buf[:96]) == bytes(8) + bytes(range(64)) + bytes(24)
            assert not any(owner.buf[96:])
            assert calls == (
                []
                if profile == "sim"
                else [
                    ("write", 8, ctypes.addressof(ctypes.c_char.from_buffer(source, 4)), 64),
                    ("read", 8, ctypes.addressof(ctypes.c_char.from_buffer(result, 8)), 64),
                ]
            )
            assert worker._get_domain(ref).host(2).identity == state.host_buffers[2].identity
            completed.append(True)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=access, daemon=True)
    try:
        thread.start()
        thread.join(3)
        assert not thread.is_alive(), "external Host access deadlocked in serializer/control admission"
        if errors:
            raise errors[0]
        assert completed == [True]
    finally:
        if not thread.is_alive():
            worker._global_node_domains.clear()
            worker.close()
        owner.close()
        owner.unlink()


@pytest.mark.parametrize("method", ["read_bytes", "write_bytes"])
def test_native_host_owner_rejects_access_without_a_live_mapping(method):
    from _task_interface import _DomainHostMemory  # noqa: PLC0415

    with pytest.raises(ValueError, match="live logical window"):
        getattr(_DomainHostMemory(), method)(0, 0, 1)


def test_host_import_leaves_local_hbm_for_ordinary_vmm_materialization(monkeypatch):
    import simpler.worker as module  # noqa: PLC0415

    command = _exported_command()
    entry = module._L2GlobalDomain(7, 3, 0, 2, command.descriptors[0], 4096, 4096, 4096, host_enabled=True)
    store = module._L2GlobalDomainStore({7: entry})
    registry = ImportRegistry(ImportContext(DEVICE_AICPU, device_owner_instance_id=b"hbmnode2"))
    payload = module.LOCAL_IMPORT_REQUEST.pack(module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 4)
    payload += encode_domain_command(command)
    monkeypatch.setattr(module, "_open_global_domain_payload", lambda _buf: _control_payload(payload))

    calls = []

    class Chip:
        def comm_global_domain_import(self, domain_id, table):
            calls.append((domain_id, table))
            return 123

        def comm_global_domain_window(self, _domain_id, slot):
            return 4096 * (slot + 1), command.descriptors[slot].mapping_size

    try:
        for _ in range(2):
            module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
        assert calls == [(7, b"".join(descriptor.encode() for descriptor in command.descriptors))]
        assert entry.grants == module._domain_buffer_slots(command)
        changed = replace(
            command,
            backing_refs=(
                replace(
                    buffer_reference(command.backing_refs[0]), identity=CanonicalIdentity(b"changed1", 99, 1)
                ).to_descriptor(),
                command.backing_refs[1],
            ),
        )
        payload = module.LOCAL_IMPORT_REQUEST.pack(module.LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, 7, 3, 4)
        payload += encode_domain_command(changed)
        with pytest.raises(ValueError, match="changes its slice grants"):
            module._handle_ctrl_global_domain_import(Chip(), memoryview(bytearray()), store, registry)
        assert entry.grants == module._domain_buffer_slots(command) and len(calls) == 1
        for descriptor in command.backing_refs:
            with pytest.raises(KeyError, match="no buffer registered"):
                registry.require(descriptor.identity)
        for slot, host in enumerate(_host_domain_buffers(command).values(), start=2):
            assert registry.require(host.identity).base == 4096 * (slot + 1)
        local_hbm = replace(
            buffer_reference(command.backing_refs[0]),
            backend_kind=BackendKind.VMM_WINDOW,
            body=(4096).to_bytes(8, "little"),
        ).to_descriptor()
        assert registry.materialize(local_hbm).base == 4096
        assert registry.require(local_hbm.identity).descriptor == local_hbm
    finally:
        registry.close()


def test_concurrent_domain_revocations_both_drain_an_authorized_hbm_copy(monkeypatch):
    from contextlib import contextmanager  # noqa: PLC0415

    from simpler.buffer import BufferCapability, wrap_vmm_window  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    hbm = wrap_vmm_window(4096, 64, worker._owner_instance_id, 41, owner_worker_id=0)
    with worker._child_prov_lock:
        worker._record_device_alloc(hbm, domain_allocation_id=7)
    fence = worker._child_prov_worker_lock(0)
    attempts = {name: threading.Event() for name in ("first-drop", "second-drop")}
    second_boundary = threading.Event()
    completed, errors = [], []

    @contextmanager
    def observed_fence(target):
        assert target == 0
        name = threading.current_thread().name
        attempts[name].set()
        if name == "second-drop":
            second_boundary.set()
        with fence:
            yield

    def revoke():
        try:
            worker._drop_domain_allocs(7)
            completed.append(threading.current_thread().name)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        finally:
            if threading.current_thread().name == "second-drop":
                second_boundary.set()

    fence.acquire()
    with worker._child_prov_lock:
        worker._require_device_capability_locked(hbm.identity, BufferCapability.COPY_TO, 64, api="copy_to")
    monkeypatch.setattr(worker, "_child_prov_worker_lock", observed_fence)
    threads = [threading.Thread(target=revoke, name=name, daemon=True) for name in attempts]
    try:
        threads[0].start()
        assert attempts["first-drop"].wait(3)
        threads[1].start()
        assert second_boundary.wait(3)
        assert attempts["second-drop"].is_set(), "a second revoke bypassed the authorized copy's fence"
        assert not completed
    finally:
        fence.release()
        for thread in threads:
            if thread.ident is not None:
                thread.join(3)
        worker._clear_child_prov()
        worker.close()
    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(completed) == ["first-drop", "second-drop"]


@pytest.mark.parametrize("mutation", ["valid", "changed-identity", "changed-base"])
def test_import_preserves_reserved_hbm_identity_and_backing(mutation):
    import simpler.worker as module  # noqa: PLC0415

    worker = module.Worker(level=3, device_ids=(0,), num_sub_workers=0)
    identity = worker._burn_buffer_identity()
    command = _exported_command()
    local = module._domain_backing_descriptor(identity, command.buffers[0].nbytes)
    command = replace(command, backing_refs=(local, command.backing_refs[1]))
    state = module._GlobalNodeDomainState(
        command=replace(command, phase=GlobalDomainPhase.PREPARE_EXPORT, descriptors=(), backing_refs=()),
        prepared_domain_ranks={0},
        descriptors={0: command.descriptors[0], 2: command.descriptors[2]},
        local_window_bases={0: 4096},
        mapping_sizes={0: 4096},
        node_worker_id=2,
        backing_ids={0: (identity,)},
        host_owner=object(),
    )
    worker._global_node_domains[7] = state
    calls = []

    class Chip:
        def control_payload(self, _kind, chip, control, _payload, _timeout):
            calls.append((chip, control))
            assert control == module.CTRL_GLOBAL_DOMAIN_IMPORT
            return module.LOCAL_IMPORT_REPLY.pack(
                module.LOCAL_DOMAIN_MAGIC,
                GLOBAL_DOMAIN_VERSION,
                7,
                3,
                123,
                8192 if mutation == "changed-base" else 4096,
                4096,
            )

    worker._worker = Chip()
    try:
        if mutation == "changed-identity":
            command = replace(
                command,
                backing_refs=(
                    replace(buffer_reference(local), identity=worker._burn_buffer_identity()).to_descriptor(),
                    command.backing_refs[1],
                ),
            )
            with pytest.raises(ValueError, match="backing references"):
                worker._import_global_domain_node(command, 2)
            assert not calls
        elif mutation == "changed-base":
            with pytest.raises(RuntimeError, match="local backing mismatch"):
                worker._import_global_domain_node(command, 2)
            assert not state.contexts
        else:
            for _ in range(2):
                worker._import_global_domain_node(command, 2)
            buffer = state.contexts[0].buffers["payload"]
            assert buffer.identity == identity and buffer.base == 4096
            assert buffer.backend_kind == BackendKind.VMM_WINDOW
            assert len(calls) == 1
    finally:
        # Synthetic grants only; no native owner or child process was created.
        worker._worker = None
        worker._global_node_domains.clear()
        worker._clear_child_prov()
        worker.close()
