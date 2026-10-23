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
from dataclasses import FrozenInstanceError, replace
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest
from simpler.buffer import AccessMode, AddressSpace, BackendKind, BufferDescriptor, CanonicalIdentity
from simpler.comm_domain import DomainConfig, domain_buffers, host_buffer
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
    decode_domain_exports,
    decode_release_command,
    encode_domain_command,
    encode_domain_exports,
    encode_release_command,
)
from simpler.task_interface import HostWindowSpec


def _command():
    return GlobalDomainCommand(
        GlobalDomainPhase.PREPARE_EXPORT,
        7,
        3,
        "mixed",
        "sim",
        4096,
        (GlobalDomainMember(2, 0, 0, 0), GlobalDomainMember(5, 0, 1, 1)),
        (GlobalDomainBuffer("payload", 32),),
        owner_instance_id=b"creator1",
        buffer_ids=(10, 11),
        host_windows=(
            GlobalDomainHostWindow(2, HostWindowSpec(0, 96), 12),
            GlobalDomainHostWindow(5, HostWindowSpec(1, 128), 13),
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
        hosts = tuple(replace(host, mapping_size=4096, handle=b"shared") for host in command.host_windows)
        descriptors = tuple(
            GlobalDomainDescriptor(GLOBAL_DOMAIN_VERSION, 1, rank, 2, 4096, b"hbm") for rank in range(2)
        )
        command = replace(command, phase=phase, descriptors=descriptors, host_windows=hosts)
    received = decode_domain_command(encode_domain_command(command))
    assert received == command
    logical, mapped = host_buffer(command, 5), host_buffer(received, 5, base=8192)
    assert logical.to_descriptor() == mapped.to_descriptor() and mapped.base == 8192
    assert struct.unpack("<QQIQ", logical.to_descriptor().body) == (7, 3, 3, 0)
    assert logical.identity != domain_buffers(command, 1)["payload"].identity
    with pytest.raises(KeyError):
        host_buffer(command, 99)
    hosts = tuple(replace(host, mapping_size=4096, handle=b"shared") for host in command.host_windows)
    exports = encode_domain_exports(command.descriptors, hosts)
    assert decode_domain_exports(exports) == (command.descriptors, hosts)
    with pytest.raises(ValueError, match="trailing"):
        decode_domain_exports(exports + b"x")


@pytest.mark.parametrize("mutation", ["order", "collision", "missing", "unprepared", "creator"])
def test_host_wire_rejects_conflicting_or_incomplete_metadata(mutation):
    command = _command()
    if mutation == "order":
        command = replace(command, host_windows=command.host_windows[::-1])
    elif mutation == "collision":
        command = replace(
            command, host_windows=(replace(command.host_windows[0], buffer_id=10), command.host_windows[1])
        )
    elif mutation == "missing":
        command = replace(command, host_windows=command.host_windows[:1])
    elif mutation == "unprepared":
        command = replace(
            command, host_windows=(replace(command.host_windows[0], handle=b"unexpected"), command.host_windows[1])
        )
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
    with pytest.raises(ValueError, match="extension"):
        decode_domain_command(encoded.replace(b"SDB2", b"SDB1"))


@pytest.mark.parametrize("slot", [0, 63, 64, 127, 128])
def test_window_slot_bound_is_enforced_by_native_descriptor(slot):
    args = (
        CanonicalIdentity(b"creator1", 9, 1),
        AddressSpace.DEVICE,
        AccessMode.READWRITE,
        BackendKind.DOMAIN_WINDOW,
        32,
        struct.pack("<QQIQ", 7, 3, slot, 0),
    )
    if slot == 128:
        with pytest.raises(ValueError, match="slot"):
            BufferDescriptor(*args)
    else:
        assert BufferDescriptor(*args).body == args[-1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_worker_id", -1),
        ("buffer_id", 0),
        ("mapping_size", 0),
        ("mapping_size", 2**64),
        ("handle", b""),
    ],
)
def test_exports_reject_invalid_host_record(field, value):
    host = replace(_command().host_windows[0], mapping_size=4096, handle=b"shared")
    with pytest.raises(ValueError):
        encode_domain_exports((), (replace(host, **{field: value}),))


@pytest.mark.parametrize(
    "node,identity,mapping,handle",
    [
        (-1, 12, 4096, b"shared"),
        (2, 0, 4096, b"shared"),
        (2, 12, 0, b"shared"),
        (2, 12, 4096, b""),
    ],
)
def test_exports_reject_invalid_host_wire(node, identity, mapping, handle):
    wire = struct.pack("<IIiiQQQI", 0, 1, node, 0, 96, identity, mapping, len(handle)) + handle
    with pytest.raises(ValueError):
        decode_domain_exports(wire)


def test_exports_reject_duplicate_and_excess_hosts():
    host = replace(_command().host_windows[0], mapping_size=4096, handle=b"shared")
    for hosts in [(host, host), tuple(replace(host, node_worker_id=i, buffer_id=i + 1) for i in range(65))]:
        with pytest.raises(ValueError):
            encode_domain_exports((), hosts)


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
        "import": [u64, ptr, size, ptr, size, ptr],
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
        rc = getattr(lib, "comm_global_domain_import")(domain_id, descriptor, 1, host, 1, ctypes.byref(context))
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
