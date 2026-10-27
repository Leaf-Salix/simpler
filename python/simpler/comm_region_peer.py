# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Authorized same-node Region imports in the consuming chip process."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from _task_interface import (
    AccessMode,
    WorkerType,
    _decode_buffer_descriptor_wire,
    _encode_buffer_descriptor_wire,
    _region_enable_peer_access,
    _region_vmm_granularity,
    _worker_host_mapped_region_import_onboard,
    _worker_host_mapped_region_import_sim,
    _worker_host_mapped_region_mapped_base,
    _worker_host_mapped_region_peek_cleanup_error,
)

from .buffer import BackendKind
from .comm_provider import (
    RegionPartKind,
    _align_up,
    _posix_object_size,
    _posix_token_from_descriptor,
    _vmm_shareable_facts,
)
from .comm_region import _close_native_lease
from .worker_chip_orch_comm import _REGION_MAGIC_VERSION, worker_chip_orch_region_desc_from_local_views

CTRL_REGION_PEER = 29
# session, transaction, resource, part, owner device, target device, operation, local VA
_PEER_HEADER = struct.Struct("<8sQQIIIIQ")
_PEER_IMPORT, _PEER_CLOSE, _PEER_ENABLE = range(3)
_DESCRIPTOR_BYTES = 88


def import_region_part(descriptor, *, part, expected, owner_device_id, target_device_id, owner_token):
    if descriptor.backend_kind is not expected:
        raise RuntimeError("create_worker_chip_region: unsupported import backend")
    selected = None if part is None else RegionPartKind(part)
    if expected is BackendKind.POSIX_SHM:
        token = _posix_token_from_descriptor(descriptor)
        mapping_bytes = _posix_object_size(token)
        if mapping_bytes < int(descriptor.nbytes):
            raise RuntimeError("POSIX shm object is shorter than the descriptor logical bytes")
        if selected is RegionPartKind.COUNTER and mapping_bytes < _align_up(int(descriptor.nbytes), 64):
            raise RuntimeError("COUNTER POSIX shm object is shorter than the 64-byte padded span")
        return _worker_host_mapped_region_import_sim(token, int(mapping_bytes), owner_token)
    if expected is BackendKind.VMM_SHAREABLE:
        device_id, shareable_handle, mapping_bytes = _vmm_shareable_facts(descriptor)
        namespace_id = owner_device_id
        if int(device_id) != int(namespace_id):
            raise RuntimeError("committed VMM device_id is outside this worker's device namespace")
        try:
            granularity = int(_region_vmm_granularity(int(device_id)))
        except Exception as exc:
            raise RuntimeError(str(exc) or "VMM granularity query failed") from exc
        if granularity < 1 or int(mapping_bytes) % granularity != 0:
            raise RuntimeError("VMM mapping_bytes must be a positive multiple of runtime granularity")
        if selected is RegionPartKind.COUNTER and int(mapping_bytes) < _align_up(int(descriptor.nbytes), 64):
            raise RuntimeError("COUNTER VMM mapping_bytes must cover 64-byte alignment")
        return _worker_host_mapped_region_import_onboard(
            int(target_device_id),
            int(shareable_handle),
            int(mapping_bytes),
            owner_token,
        )
    raise RuntimeError("create_worker_chip_region: unsupported import backend")


def region_binding_prefix(args):
    if args.scalar_count() == 0 or int(args.scalar(0)) != _REGION_MAGIC_VERSION:
        return None
    if args.scalar_count() < 6:
        raise ValueError("Region binding requires six scalars")
    return tuple(int(args.scalar(i)) for i in range(6))


@dataclass
class _PeerImport:
    request: bytes
    lease: Any = None
    error: BaseException | None = None
    closed: bool = False
    base: int = 0

    def close(self):
        if self.error is not None:
            raise self.error
        if not self.closed:
            try:
                if self.lease is not None:
                    _close_native_lease(self.lease)
            except BaseException as exc:
                self.error = exc
                raise
            self.closed = True
            self.lease = None


class PeerRegionImports:
    def __init__(self, device_id, platform, owner_token):
        self.device_id = device_id
        self.platform = platform
        self.owner_token = owner_token
        self.records = {}

    @property
    def holds_resources(self):
        return any(not record.closed for record in self.records.values())

    def handle(self, frame):
        if len(frame) != _PEER_HEADER.size + _DESCRIPTOR_BYTES:
            raise ValueError("invalid Region peer control frame size")
        session, transaction, resource, part, owner, target, operation, base = _PEER_HEADER.unpack_from(frame)
        if target != self.device_id or owner == target or transaction == 0 or resource == 0 or base != 0:
            raise ValueError("invalid Region peer target or allocation identity")
        RegionPartKind(part)
        if operation == _PEER_ENABLE:
            self._enable(owner)
            return
        key = session, transaction, part
        request = bytes(frame[_PEER_HEADER.size :])
        identity = _PEER_HEADER.pack(session, transaction, resource, part, owner, target, _PEER_IMPORT, 0) + request
        record = self.records.get(key)
        self._validate_import_identity(key, identity, resource, owner, target, request)
        if operation == _PEER_CLOSE:
            if record is None:
                record = self.records[key] = _PeerImport(identity)
            record.close()
            return
        if operation != _PEER_IMPORT:
            raise ValueError("invalid Region peer operation")
        if record is not None:
            if record.error is not None:
                raise record.error
            if record.closed:
                raise ValueError("Region peer import is already closed")
        else:
            descriptor = _decode_buffer_descriptor_wire(request)
            if descriptor.access is not AccessMode.READWRITE or int(descriptor.identity.generation) != 1:
                raise ValueError("Region peer descriptor requires READWRITE and generation one")
            record = self.records[key] = _PeerImport(identity)
            try:
                self._enable(owner)
                record.lease = import_region_part(
                    descriptor,
                    part=part,
                    expected=BackendKind.POSIX_SHM if self.platform.endswith("sim") else BackendKind.VMM_SHAREABLE,
                    owner_device_id=owner,
                    target_device_id=target,
                    owner_token=self.owner_token,
                )
                record.base = int(_worker_host_mapped_region_mapped_base(int(record.lease)))
            except BaseException as exc:
                debt = _worker_host_mapped_region_peek_cleanup_error(self.owner_token)
                if debt or record.lease is not None:
                    record.error = exc
                else:
                    record.closed = True
                raise
        struct.pack_into("<Q", frame, _PEER_HEADER.size - 8, record.base)

    def _validate_import_identity(self, key, identity, resource, owner, target, request):
        session, transaction, part = key
        record = self.records.get(key)
        for (other_session, other_transaction, other_part), other in self.records.items():
            if (other_session, other_transaction) == (session, transaction) and other_part != part:
                fields = _PEER_HEADER.unpack_from(other.request)
                if (fields[2], fields[4], fields[5]) != (resource, owner, target):
                    raise ValueError("Region peer parts name different allocations")
                if (
                    _decode_buffer_descriptor_wire(other.request[_PEER_HEADER.size :]).identity.owner_instance_id
                    != _decode_buffer_descriptor_wire(request).identity.owner_instance_id
                ):
                    raise ValueError("Region peer parts name different owners")
        if record is not None and record.request != identity:
            raise ValueError("Region peer allocation conflicts with existing import")

    def _enable(self, owner):
        if not self.platform.endswith("sim"):
            _region_enable_peer_access(self.device_id, owner)

    def validates(self, binding):
        for (session, transaction, part), record in self.records.items():
            if part != int(RegionPartKind.PAYLOAD) or record.closed or record.error is not None:
                continue
            counter = self.records.get((session, transaction, int(RegionPartKind.COUNTER)))
            if counter is None or counter.closed or counter.error is not None:
                continue
            resource = _PEER_HEADER.unpack_from(record.request)[2]
            payload_desc = _decode_buffer_descriptor_wire(record.request[_PEER_HEADER.size :])
            counter_desc = _decode_buffer_descriptor_wire(counter.request[_PEER_HEADER.size :])
            if binding == (
                _REGION_MAGIC_VERSION,
                resource,
                record.base,
                payload_desc.nbytes,
                counter.base,
                counter_desc.nbytes,
            ):
                return True
        return False

    def validate_args(self, args, provider_store):
        binding = region_binding_prefix(args)
        if binding is None or self.validates(binding):
            return
        views = [
            provider_store.local_view(binding[1], part) for part in (RegionPartKind.PAYLOAD, RegionPartKind.COUNTER)
        ]
        if binding != tuple(worker_chip_orch_region_desc_from_local_views(binding[1], *views).scalars()):
            raise ValueError("Region binding does not belong to this chip")

    def close(self):
        first = None
        for record in self.records.values():
            try:
                record.close()
            except BaseException as exc:
                first = first or exc
        if first is not None:
            raise first


class PeerRegionLease:
    def __init__(self, worker, instance, descriptor, part, consumer_worker_id, owner_device_id):
        self.worker = worker
        self.consumer_worker_id = consumer_worker_id
        self.mapped_base = 0
        self.closed = False
        self.error = None
        target = worker._provider_import_device_id(consumer_worker_id)
        session, transaction = instance._allocation_identity
        self.frame = _PEER_HEADER.pack(
            session, transaction, instance.provider_resource_id, int(part), owner_device_id, target, _PEER_IMPORT, 0
        ) + bytes(_encode_buffer_descriptor_wire(descriptor))

    def exchange(self, operation):
        frame = bytearray(self.frame)
        struct.pack_into("<I", frame, _PEER_HEADER.size - 12, operation)
        reply = bytes(
            self.worker._worker.control_payload(
                WorkerType.NEXT_LEVEL,
                self.consumer_worker_id,
                CTRL_REGION_PEER,
                frame,
                self.worker._py_control_timeout_s,
            )
        )
        if (
            len(reply) != len(frame)
            or reply[: -_DESCRIPTOR_BYTES - 8] != frame[: -_DESCRIPTOR_BYTES - 8]
            or reply[-_DESCRIPTOR_BYTES:] != frame[-_DESCRIPTOR_BYTES:]
        ):
            raise RuntimeError("Region peer reply identity mismatch")
        return struct.unpack_from("<Q", reply, _PEER_HEADER.size - 8)[0]

    def close(self):
        if self.error is not None:
            raise self.error
        if not self.closed:
            try:
                self.exchange(_PEER_CLOSE)
            except BaseException as exc:
                self.error = exc
                raise
            self.closed = True
