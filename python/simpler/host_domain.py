# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local Host backing and fixed per-chip attachments owned by a CommDomain."""

from __future__ import annotations

import ctypes

from _task_interface import _encode_buffer_descriptor_wire

from .buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    Tensor,
)
from .comm_domain import HostWindowSpec, domain_mapping_ref, domain_mapping_target, require_domain_mapping_access
from .host_domain_backend import host_backing_type


class HostDomainResource:
    """Owner retained until all member views are confirmed gone.

    Attachments are fixed for the domain lifetime. A new domain burns a new backing
    identity; a selector cannot survive release even if its name, rank or VA repeats.
    """

    def __init__(self, spec: HostWindowSpec, identity, workers, platform, device):
        self.spec = spec
        self.identity = identity
        self.workers = tuple(workers)
        self.device = device
        self.backend = host_backing_type(platform)()
        self.buffer = None
        self._descriptor = None
        self.export = None
        self.published = False
        self.views_released = False
        self.closed = False

    def allocate(self):
        base, self.export = self.backend.allocate(self.spec, self.identity, self.device, len(self.workers))
        self.buffer = Buffer(
            identity=self.identity,
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.REMOTE_SIDECAR,
            nbytes=self.spec.nbytes,
            base=base,
        )
        self._descriptor = self.buffer.to_descriptor()

    def request(self) -> bytes:
        if self.export is None or self.buffer is None:
            raise RuntimeError("Host backing has not been allocated")
        return self.export.encode() + _encode_buffer_descriptor_wire(self._descriptor)

    def map_view(self, tensor: Tensor, worker: int, *, access=AccessMode.READWRITE) -> Tensor:
        if self.closed or self.buffer is None or self.buffer.closed or tensor.buffer != self._descriptor:
            raise ValueError("Host mapping requires the live domain backing")
        if tensor.mapping_ref:
            raise ValueError("map_view requires the native Tensor view")
        return tensor._with_mapping_ref(domain_mapping_ref(self.workers.index(worker), access))

    def validate(self, tensor, worker, *, native=False, required=0):
        if self.closed or self.buffer is None or self.buffer.closed or tensor.buffer != self._descriptor:
            raise ValueError("Host domain Tensor is stale or differs from its source registration")
        if worker not in self.workers or (
            not (native and tensor.mapping_ref == 0)
            and (not tensor.mapping_ref or domain_mapping_target(tensor.mapping_ref)[0] != self.workers.index(worker))
        ):
            raise ValueError("Host Tensor requires an explicit mapping for the target worker")
        if tensor.mapping_ref:
            require_domain_mapping_access(tensor.mapping_ref, required)

    def copy(self, offset, value, *, writing):
        size = len(value) if writing else value
        if offset < 0 or size < 0 or offset > self.spec.nbytes or size > self.spec.nbytes - offset:
            raise ValueError("Host domain copy exceeds the backing range")
        if self.closed or self.buffer is None or self.buffer.closed:
            raise ValueError("Host domain backing is not live")
        base = self.buffer.base
        if writing:
            ctypes.memmove(base + offset, value, size)
            return None
        return ctypes.string_at(base + offset, size)

    def close(self):
        self.closed = True
        if self.buffer is not None:
            self.buffer.closed = True
        if self.published and not self.views_released:
            raise RuntimeError("Host domain owner retained: member view release is unconfirmed")
        self.backend.release()
