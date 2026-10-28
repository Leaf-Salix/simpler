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
from dataclasses import dataclass, field
from typing import Any

from _task_interface import (
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
    RegionExportDescriptor,
    RegionPartKind,
    RegionPartLocalView,
    _align_up,
    _posix_object_size,
    _posix_token_from_descriptor,
    _vmm_shareable_facts,
)
from .comm_provider_control import _validate_decoded_descriptor_pair
from .comm_region import _close_native_lease
from .worker_chip_orch_comm import _REGION_MAGIC_VERSION, worker_chip_orch_region_desc_from_local_views

CTRL_REGION_PEER = 29
# session, transaction, resource, owner device, target device, operation
_PEER_HEADER = struct.Struct("<8sQQIII")
_PEER_BASES = struct.Struct("<QQ")
_PEER_IMPORT, _PEER_CLOSE, _PEER_ENABLE, _PEER_CONFIRM = range(4)
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
    leases: list[Any] = field(default_factory=list)
    binding: tuple | None = None
    error: BaseException | None = None
    closed: bool = False
    close_attempted: bool = False

    def close(self):
        if not self.close_attempted:
            self.close_attempted = True
            for lease in self.leases:
                try:
                    _close_native_lease(lease)
                except BaseException as exc:
                    self.error = self.error or exc
        if self.error is not None:
            raise self.error
        self.closed = True
        self.leases.clear()


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
        if len(frame) != _PEER_HEADER.size + 2 * _DESCRIPTOR_BYTES + _PEER_BASES.size:
            raise ValueError("invalid Region peer control frame size")
        session, transaction, resource, owner, target, operation = _PEER_HEADER.unpack_from(frame)
        if target != self.device_id or owner == target or transaction == 0 or resource == 0:
            raise ValueError("invalid Region peer target or allocation identity")
        if any(_PEER_BASES.unpack_from(frame, len(frame) - _PEER_BASES.size)):
            raise ValueError("Region peer request must not contain local addresses")
        if operation in (_PEER_ENABLE, _PEER_CONFIRM):
            self._enable(owner, wait=operation == _PEER_CONFIRM)
            return
        key = session, transaction
        request = bytes(frame[_PEER_HEADER.size : -_PEER_BASES.size])
        identity = _PEER_HEADER.pack(session, transaction, resource, owner, target, _PEER_IMPORT) + request
        record = self.records.get(key)
        if record is not None and record.request != identity:
            raise ValueError("Region peer allocation conflicts with existing import")
        if operation == _PEER_CLOSE:
            self.records.setdefault(key, _PeerImport(identity)).close()
            return
        if operation != _PEER_IMPORT:
            raise ValueError("invalid Region peer operation")
        if record is not None:
            if record.error is not None:
                raise record.error
            if record.closed or record.binding is None:
                raise ValueError("Region peer import is closed or incomplete")
        else:
            descriptor = RegionExportDescriptor(
                _decode_buffer_descriptor_wire(request[:_DESCRIPTOR_BYTES]),
                _decode_buffer_descriptor_wire(request[_DESCRIPTOR_BYTES:]),
            )
            _validate_decoded_descriptor_pair(descriptor)
            record = self.records[key] = _PeerImport(identity)
            try:
                self._enable(owner)
                views = []
                for part, desc in (
                    (RegionPartKind.PAYLOAD, descriptor.payload),
                    (RegionPartKind.COUNTER, descriptor.counter),
                ):
                    lease = import_region_part(
                        desc,
                        part=part,
                        expected=BackendKind.POSIX_SHM if self.platform.endswith("sim") else BackendKind.VMM_SHAREABLE,
                        owner_device_id=owner,
                        target_device_id=target,
                        owner_token=self.owner_token,
                    )
                    record.leases.append(lease)
                    base = int(_worker_host_mapped_region_mapped_base(int(lease)))
                    if base <= 0:
                        raise RuntimeError("Region peer import returned no local mapping")
                    views.append(RegionPartLocalView(part, base, int(desc.nbytes)))
                record.binding = tuple(worker_chip_orch_region_desc_from_local_views(resource, *views).scalars())
            except BaseException as exc:
                if _worker_host_mapped_region_peek_cleanup_error(self.owner_token):
                    record.error = exc
                raise
        _PEER_BASES.pack_into(frame, len(frame) - _PEER_BASES.size, record.binding[2], record.binding[4])

    def _enable(self, owner, *, wait=True):
        if not self.platform.endswith("sim"):
            _region_enable_peer_access(self.device_id, owner, wait)

    def validates(self, binding):
        return any(
            not record.closed and record.error is None and record.binding == binding for record in self.records.values()
        )

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
    def __init__(self, worker, instance, descriptor, consumer_worker_id, owner_device_id):
        self.worker = worker
        self.owner_worker_id = instance.worker_id
        self.descriptor = descriptor
        self.consumer_worker_id = consumer_worker_id
        self.closed = False
        self.error = None
        target = worker._provider_import_device_id(consumer_worker_id)
        session, transaction = instance._allocation_identity
        self.header = (session, transaction, instance.provider_resource_id, owner_device_id, target)
        self.descriptors = bytes(_encode_buffer_descriptor_wire(descriptor.payload)) + bytes(
            _encode_buffer_descriptor_wire(descriptor.counter)
        )

    def exchange(self, operation):
        session, transaction, resource, owner, target = self.header
        worker_id = self.consumer_worker_id
        if operation in (_PEER_ENABLE, _PEER_CONFIRM):
            owner, target = target, owner
            worker_id = self.owner_worker_id
        frame = bytearray(_PEER_HEADER.pack(session, transaction, resource, owner, target, operation))
        frame += self.descriptors + bytes(_PEER_BASES.size)
        expected_prefix = bytes(frame[: -_PEER_BASES.size])
        reply = bytes(
            self.worker._worker.control_payload(
                WorkerType.NEXT_LEVEL, worker_id, CTRL_REGION_PEER, frame, self.worker._py_control_timeout_s
            )
        )
        if len(reply) != len(frame) or reply[: -_PEER_BASES.size] != expected_prefix:
            raise RuntimeError("Region peer reply identity mismatch")
        return _PEER_BASES.unpack_from(reply, len(reply) - _PEER_BASES.size)

    def import_views(self):
        self.exchange(_PEER_ENABLE)
        bases = self.exchange(_PEER_IMPORT)
        self.exchange(_PEER_CONFIRM)
        if any(base <= 0 for base in bases):
            raise RuntimeError("Region peer import returned no local mapping")
        return (
            RegionPartLocalView(RegionPartKind.PAYLOAD, bases[0], int(self.descriptor.payload.nbytes)),
            RegionPartLocalView(RegionPartKind.COUNTER, bases[1], int(self.descriptor.counter.nbytes)),
        )

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
