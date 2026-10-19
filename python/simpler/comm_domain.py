# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Communication-domain configuration snapshots, creator registry and address-free window slices.

One registry per Worker owns its domains' names, refs and creation transactions whatever backend
built them (L3-local windows or an L4-brokered Global CommDomain). A window slice is named by an
address-free ``DOMAIN_WINDOW`` descriptor; only the chip that holds the rank's mapping resolves it.
"""

from __future__ import annotations

import operator
import struct
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    CommDomainRef,
)

from .buffer import Buffer
from .global_comm_domain import GLOBAL_DOMAIN_MAX_RANKS, CommBufferDefinition

if TYPE_CHECKING:
    from .task_interface import CommDomainHandle

# DomainWindowBody in buffer.h: creator nonce, domain_id, generation, domain_rank, byte offset.
DOMAIN_WINDOW_BODY = struct.Struct("<8sQIIQ")


def domain_window_location(descriptor: BufferDescriptor) -> tuple[CommDomainRef, int, int]:
    """``(ref, domain_rank, offset)`` of the slice a ``DOMAIN_WINDOW`` descriptor names."""
    creator, domain_id, generation, rank, offset = DOMAIN_WINDOW_BODY.unpack(bytes(descriptor.body))
    return CommDomainRef(creator, domain_id, generation), rank, offset


def domain_slices(
    ref: CommDomainRef, buffers: tuple[CommBufferDefinition, ...], rank: int, buffer_ids: tuple[int, ...]
) -> Mapping[str, Buffer]:
    """One rank's address-free slice Buffers, in window order, under the creator's identities."""
    slices: dict[str, Buffer] = {}
    offset = 0
    for spec, buffer_id in zip(buffers, buffer_ids):
        slices[spec.name] = Buffer(
            identity=CanonicalIdentity(ref.owner_instance_id, int(buffer_id), 1),
            address_space=AddressSpace.DEVICE,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.DOMAIN_WINDOW,
            nbytes=spec.nbytes,
            body=DOMAIN_WINDOW_BODY.pack(ref.owner_instance_id, ref.domain_id, ref.generation, rank, offset),
        )
        offset += spec.nbytes
    return MappingProxyType(slices)


def domain_slice_rank(handle: CommDomainHandle, descriptor: BufferDescriptor) -> int:
    """The domain rank of a slice descriptor that is exactly one of ``handle``'s own."""
    ref, rank, _offset = domain_window_location(descriptor)
    if ref != handle.ref or not 0 <= rank < len(handle.members):
        raise ValueError(f"domain tensor does not name a rank of domain {handle.name!r}")
    if descriptor not in [buffer.to_descriptor() for buffer in handle.members[rank].buffers.values()]:
        raise ValueError(f"domain tensor does not match the canonical slice catalog of domain {handle.name!r}")
    return rank


@dataclass(frozen=True)
class DomainConfig:
    """Immutable snapshot of everything that shapes a domain; reuse requires every field to match."""

    members: tuple[tuple[Any, ...], ...]
    window_size: int
    buffers: tuple[CommBufferDefinition, ...]
    retain_after_run: bool
    topology: tuple[Any, ...]

    @classmethod
    def capture(cls, *, members, window_size, buffers, retain_after_run, topology) -> DomainConfig:
        members = tuple(tuple(member) for member in members)
        window_size = operator.index(window_size)
        specs = tuple(
            CommBufferDefinition(
                str(spec.name),
                str(spec.dtype),
                operator.index(spec.count),
                operator.index(spec.nbytes),
                bool(spec.load_from_host),
                bool(spec.store_to_host),
            )
            for spec in buffers
        )
        if not members or len(members) > GLOBAL_DOMAIN_MAX_RANKS:
            raise ValueError(f"domain members must contain between 1 and {GLOBAL_DOMAIN_MAX_RANKS} chips")
        if len(set(members)) != len(members):
            raise ValueError("domain members contain a duplicate chip")
        if window_size <= 0:
            raise ValueError("domain window_size must be positive")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError(f"domain buffer names must be unique: {[spec.name for spec in specs]}")
        if any(not spec.name or spec.nbytes <= 0 for spec in specs):
            raise ValueError("domain buffers require a name and positive nbytes")
        total = sum(spec.nbytes for spec in specs)
        if total > window_size:
            raise ValueError(f"domain buffers sum to {total} bytes, exceeds window_size={window_size}")
        return cls(members, window_size, specs, bool(retain_after_run), tuple(topology))

    def difference(self, other: DomainConfig) -> str:
        """The first field that differs, named precisely enough to fix the call; empty if none."""
        for item in fields(self):
            before, after = getattr(self, item.name), getattr(other, item.name)
            if before == after:
                continue
            if item.name == "buffers" and len(before) == len(after):
                for index, (old, new) in enumerate(zip(before, after)):
                    for part in fields(old):
                        was, now = getattr(old, part.name), getattr(new, part.name)
                        if was != now:
                            return f"buffers[{index}].{part.name}: {was!r} != {now!r}"
            return f"{item.name}: {before!r} != {after!r}"
        return ""


@dataclass(eq=False)
class _Creation:
    config: DomainConfig
    done: bool = False
    error: BaseException | None = None


class CommDomainRegistry:
    """Names, refs and creation transactions of the domains one Worker created.

    A name is published only once every member committed; a concurrent request with the same
    configuration waits for that one transaction, a different one is refused. Release unpublishes
    the name at once so it can be rebuilt, while the ref stays resolvable — and refused — until the
    backend windows are freed.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.RLock())
        self._names: dict[str, CommDomainHandle] = {}
        self._creating: dict[str, _Creation] = {}
        self._refs: dict[CommDomainRef, tuple[CommDomainHandle, DomainConfig]] = {}

    def acquire(
        self,
        name: str,
        config: DomainConfig,
        *,
        reuse: bool,
        create: Callable[[], CommDomainHandle],
        use: Callable[[CommDomainHandle], None],
    ) -> CommDomainHandle:
        if not name:
            raise ValueError("domain name must be non-empty")
        with self._cv:
            while True:
                live = self._names.get(name)
                if live is not None:
                    if not reuse:
                        raise ValueError(f"domain {name!r} is already live")
                    difference = self._refs[live.ref][1].difference(config)
                    if difference:
                        raise ValueError(f"domain {name!r} configuration differs: {difference}")
                    use(live)
                    return live
                pending = self._creating.get(name)
                if pending is None:
                    break
                if not reuse:
                    raise ValueError(f"domain {name!r} is already being created")
                difference = pending.config.difference(config)
                if difference:
                    raise ValueError(f"domain {name!r} configuration differs: {difference}")
                self._cv.wait_for(lambda pending=pending: pending.done)
                if pending.error is not None:
                    raise pending.error
            creation = _Creation(config)
            self._creating[name] = creation
        handle = None
        try:
            handle = create()
            with self._cv:
                self._names[name] = handle
                self._refs[handle.ref] = (handle, config)
        except BaseException as exc:
            creation.error = exc
            raise
        finally:
            with self._cv:
                creation.done = True
                if self._creating.get(name) is creation:
                    del self._creating[name]
                self._cv.notify_all()
        return handle

    def lookup(self, key: str | CommDomainRef) -> CommDomainHandle | None:
        with self._cv:
            if isinstance(key, str):
                return self._names.get(key)
            entry = self._refs.get(key)
            return None if entry is None else entry[0]

    def unpublish(self, handle: CommDomainHandle) -> None:
        with self._cv:
            if self._names.get(handle.name) is handle:
                del self._names[handle.name]

    def retire(self, handle: CommDomainHandle) -> None:
        with self._cv:
            entry = self._refs.get(handle.ref)
            if entry is not None and entry[0] is handle:
                del self._refs[handle.ref]

    def live(self) -> dict[str, CommDomainHandle]:
        with self._cv:
            return dict(self._names)


@dataclass(frozen=True)
class ChipDomainMapping:
    generation: int
    domain_rank: int
    local_window_base: int
    mapping_size: int
    device_ctx: int
    # Window offset -> (canonical identity, nbytes) of the slice that starts there.
    slices: Mapping[int, tuple[CanonicalIdentity, int]]


class ChipDomainStore:
    """The domain windows one chip maps, and the domains its current task may resolve.

    A slice resolves only when the task names its domain, the slice belongs to this chip's rank,
    its identity is the catalog's for that offset, and it lies inside the actual mapping.
    """

    def __init__(self) -> None:
        self._mappings: dict[CommDomainRef, ChipDomainMapping] = {}
        self.task_refs: frozenset[CommDomainRef] = frozenset()

    def install(self, ref: CommDomainRef, mapping: ChipDomainMapping) -> None:
        self._mappings[ref] = mapping

    def revoke(self, ref: CommDomainRef) -> None:
        self._mappings.pop(ref, None)

    def mapping(self, ref: CommDomainRef) -> ChipDomainMapping | None:
        return self._mappings.get(ref)

    def resolve(self, descriptor: BufferDescriptor) -> int:
        ref, rank, offset = domain_window_location(descriptor)
        if ref not in self.task_refs:
            raise RuntimeError("domain tensor names a domain its task was not accepted to use")
        mapping = self._mappings.get(ref)
        if mapping is None:
            raise RuntimeError("domain tensor names a domain this chip does not map")
        if rank != mapping.domain_rank:
            raise ValueError(f"domain tensor names rank {rank}; this chip maps rank {mapping.domain_rank}")
        if mapping.slices.get(offset) != (descriptor.identity, descriptor.nbytes):
            raise ValueError("domain tensor does not match this chip's canonical slice catalog")
        if offset > mapping.mapping_size or descriptor.nbytes > mapping.mapping_size - offset:
            raise ValueError("domain tensor exceeds the actual window mapping")
        return mapping.local_window_base + offset
