# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Communication-domain identities, immutable configuration and fenced ownership."""

from __future__ import annotations

import json
import operator
import struct
import threading
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any

from _task_interface import (
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CommDomainRef,
    _decode_buffer_descriptor_wire,
    _encode_buffer_descriptor_wire,
)

from .buffer import Buffer

_WINDOW_BODY = struct.Struct("<8sQIIQ")
_MAX_METADATA_BYTES = 4 * 1024 * 1024


@dataclass
class CommBufferSpec:
    """A named, sequentially placed slice of every rank's window."""

    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool = False
    store_to_host: bool = False


@dataclass(frozen=True)
class _BufferDefinition:
    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool
    store_to_host: bool


@dataclass(frozen=True)
class _DomainMember:
    node_worker_id: int | None
    local_worker_id: int
    global_device_rank: int
    target_identity: Hashable


@dataclass(frozen=True)
class _DomainConfig:
    members: tuple[_DomainMember, ...]
    window_size: int
    buffers: tuple[_BufferDefinition, ...]
    retain_after_run: bool
    topology: tuple

    @classmethod
    def capture(cls, *, members, window_size, buffers, retain_after_run, topology):
        window_size = operator.index(window_size)
        members = tuple(members)
        definitions = tuple(
            _BufferDefinition(
                spec.name,
                spec.dtype,
                operator.index(spec.count),
                operator.index(spec.nbytes),
                bool(spec.load_from_host),
                bool(spec.store_to_host),
            )
            for spec in buffers
        )
        if not members or len(set((m.node_worker_id, m.local_worker_id) for m in members)) != len(members):
            raise ValueError("members must be non-empty and contain no duplicate targets")
        if any(m.local_worker_id < 0 for m in members):
            raise ValueError("members must name non-negative chip worker ids")
        if not 0 < window_size < (1 << 64):
            raise ValueError("window_size must be a positive uint64")
        if any(
            not isinstance(spec.name, str)
            or not spec.name
            or not isinstance(spec.dtype, str)
            or not spec.dtype
            or spec.count <= 0
            or spec.nbytes <= 0
            for spec in definitions
        ):
            raise ValueError("buffers require non-empty names/dtypes and positive count/nbytes")
        if len({spec.name for spec in definitions}) != len(definitions):
            raise ValueError("buffer names must be unique")
        if sum(spec.nbytes for spec in definitions) > window_size:
            raise ValueError("buffers exceed requested window_size")
        return cls(members, window_size, definitions, bool(retain_after_run), tuple(topology))

    def difference(self, other: _DomainConfig) -> str:
        for f in fields(self):
            before, after = getattr(self, f.name), getattr(other, f.name)
            if before == after:
                continue
            if f.name == "buffers" and len(before) == len(after):
                for i, (old, new) in enumerate(zip(before, after)):
                    for part in fields(old):
                        if getattr(old, part.name) != getattr(new, part.name):
                            return (
                                f"buffers[{i}].{part.name}: {getattr(old, part.name)!r} != {getattr(new, part.name)!r}"
                            )
            return f"{f.name}: {before!r} != {after!r}"
        return ""


def _validate_catalog(ref, config, catalog):
    if len(catalog) != len(config.members) or any(len(row) != len(config.buffers) for row in catalog):
        raise ValueError("domain catalog dimensions do not match configuration")
    identities = set()
    for rank, row in enumerate(catalog):
        offset = 0
        for spec, descriptor in zip(config.buffers, row):
            if (
                descriptor.backend_kind != BackendKind.DOMAIN_WINDOW
                or descriptor.address_space != AddressSpace.DEVICE
                or descriptor.access != AccessMode.READWRITE
                or descriptor.identity.owner_instance_id != ref.owner_instance_id
                or descriptor.body
                != _WINDOW_BODY.pack(ref.owner_instance_id, ref.domain_id, ref.generation, rank, offset)
                or descriptor.nbytes != spec.nbytes
            ):
                raise ValueError("domain catalog does not match canonical ref/rank/slice")
            if descriptor.identity in identities:
                raise ValueError("domain catalog contains a duplicate canonical identity")
            identities.add(descriptor.identity)
            offset += spec.nbytes


@dataclass
class ChipDomainContext:
    name: str
    domain_rank: int
    domain_size: int
    device_ctx: int
    local_window_base: int
    actual_window_size: int
    buffers: Mapping[str, Buffer]


@dataclass(frozen=True)
class CommDomainMember:
    domain_rank: int
    node_worker_id: int | None
    local_worker_id: int
    global_device_rank: int
    target_identity: Hashable
    buffers: Mapping[str, Buffer]


@dataclass(eq=False)
class _DomainRecord:
    name: str
    ref: CommDomainRef
    config: _DomainConfig
    owner: bool = True
    state: str = "creating"
    mapping_size: int = 0
    contexts: dict[int, ChipDomainContext] = field(default_factory=dict)
    descriptors: tuple[tuple[BufferDescriptor, ...], ...] = ()
    members: tuple[CommDomainMember, ...] = ()
    backend: Any = None
    attachments: tuple = ()
    uses: set[Hashable] = field(default_factory=set)
    unsettled: set[Hashable] = field(default_factory=set)
    error: BaseException | None = None
    free_error: BaseException | None = None
    creation_done: bool = False
    freeing: object | None = None
    handle: CommDomainHandle = field(init=False, repr=False)

    def install_context(self, rank, device_ctx, base, size):
        buffers = {}
        offset = 0
        for spec in self.config.buffers:
            logical = self.members[rank].buffers[spec.name]
            local = replace(logical, base=base + offset)
            local._domain_check = logical._domain_check
            buffers[spec.name] = local
            offset += spec.nbytes
        context = ChipDomainContext(
            self.name, rank, len(self.members), device_ctx, base, size, MappingProxyType(buffers)
        )
        self.contexts[rank] = context
        return context


class CommDomainHandle:
    """One shared domain handle; borrowed handles never own collective release."""

    def __init__(self, registry: _DomainRegistry, record: _DomainRecord):
        self._registry = registry
        self._record = record

    @property
    def name(self):
        return self._record.name

    @property
    def ref(self):
        return self._record.ref

    @property
    def domain_id(self):
        return self.ref.domain_id

    @property
    def allocation_id(self):
        """Compatibility identifier; backend allocation ids remain private."""
        return self.domain_id

    @property
    def generation(self):
        return self.ref.generation

    @property
    def members(self):
        return self._record.members

    @property
    def buffers(self):
        return self._record.config.buffers

    @property
    def attachments(self):
        return self._record.attachments

    @property
    def requested_window_size(self):
        return self._record.config.window_size

    @property
    def mapping_size(self):
        return self._record.mapping_size

    @property
    def retain_after_run(self):
        return self._record.config.retain_after_run

    @property
    def released(self):
        return self._record.state in ("released", "freed")

    @property
    def freed(self):
        return self._record.state == "freed"

    @property
    def committed(self):
        """Compatibility view of the unified live state."""
        return self._record.state == "live"

    @property
    def borrowed(self):
        return not self._record.owner

    def member(self, domain_rank: int) -> CommDomainMember:
        self._registry.access(self._record)
        rank = operator.index(domain_rank)
        if rank < 0 or rank >= len(self.members):
            raise IndexError(f"domain rank {rank} is out of range")
        return self.members[rank]

    def local_context(self, domain_rank: int) -> ChipDomainContext:
        self.member(domain_rank)
        try:
            return self._record.contexts[domain_rank]
        except KeyError as exc:
            raise ValueError(f"domain rank {domain_rank} has no mapping on this node") from exc

    def buffer_range(self, name: str) -> tuple[int, int]:
        self._registry.access(self._record)
        offset = 0
        for spec in self.buffers:
            if spec.name == name:
                return offset, spec.nbytes
            offset += spec.nbytes
        raise KeyError(name)

    @property
    def workers(self):
        return tuple(self.members[rank].local_worker_id for rank in self._record.contexts)

    @property
    def contexts(self):
        self._registry.access(self._record)
        return MappingProxyType({self.members[r].local_worker_id: ctx for r, ctx in self._record.contexts.items()})

    def __getitem__(self, chip_worker_id):
        return self.contexts[chip_worker_id]

    def release(self):
        if self.borrowed:
            raise RuntimeError("borrowed domain handles cannot release the domain")
        self._registry.release(self._record)

    def __enter__(self):
        if self.borrowed:
            raise RuntimeError("borrowed domain handles cannot enter a releasing context manager")
        self._registry.access(self._record)
        return self

    def __exit__(self, *_):
        self.release()

    def __repr__(self):
        return f"CommDomainHandle(name={self.name!r}, ref={self.ref!r}, state={self._record.state!r})"


class _DomainRegistry:
    def __init__(self, *, owner_instance_id, mint_identity, allocate, destroy, current_use=None, poison=None):
        self.owner_instance_id = bytes(owner_instance_id)
        self._mint_identity = mint_identity
        self._allocate = allocate
        self._destroy = destroy
        self._current_use = current_use
        self._poison = poison
        self._cv = threading.Condition(threading.RLock())
        self._names: dict[str, _DomainRecord] = {}
        self._records: dict[CommDomainRef, _DomainRecord] = {}
        self._runs: dict[Hashable, set[CommDomainRef]] = {}
        self._next_id = 1

    @property
    def records(self):
        with self._cv:
            return dict(self._records)

    @property
    def unsettled_refs(self):
        with self._cv:
            return {ref for ref, record in self._records.items() if record.unsettled}

    def live_domains(self):
        with self._cv:
            return {name: record.handle for name, record in self._names.items() if record.state == "live"}

    def _catalog(self, record, descriptors=None):
        rows, members = [], []
        for rank, member in enumerate(record.config.members):
            row, buffers, offset = [], {}, 0
            for index, spec in enumerate(record.config.buffers):
                descriptor = (
                    descriptors[rank][index]
                    if descriptors is not None
                    else BufferDescriptor(
                        identity=self._mint_identity(),
                        address_space=AddressSpace.DEVICE,
                        access=AccessMode.READWRITE,
                        backend_kind=BackendKind.DOMAIN_WINDOW,
                        nbytes=spec.nbytes,
                        body=_WINDOW_BODY.pack(
                            record.ref.owner_instance_id, record.ref.domain_id, record.ref.generation, rank, offset
                        ),
                    )
                )
                if descriptor.body != _WINDOW_BODY.pack(
                    record.ref.owner_instance_id, record.ref.domain_id, record.ref.generation, rank, offset
                ):
                    raise ValueError("domain catalog does not match ref/rank/slice")
                if descriptor.nbytes != spec.nbytes:
                    raise ValueError("domain catalog does not match buffer extent")
                buffer = Buffer(
                    identity=descriptor.identity,
                    address_space=descriptor.address_space,
                    access=descriptor.access,
                    backend_kind=descriptor.backend_kind,
                    nbytes=descriptor.nbytes,
                    body=descriptor.body,
                    owner_worker_id=member.local_worker_id,
                )
                buffer._domain_check = lambda r=record: self.access(r)
                buffers[spec.name] = buffer
                row.append(descriptor)
                offset += spec.nbytes
            rows.append(tuple(row))
            members.append(
                CommDomainMember(
                    rank,
                    member.node_worker_id,
                    member.local_worker_id,
                    member.global_device_rank,
                    member.target_identity,
                    MappingProxyType(buffers),
                )
            )
        record.descriptors, record.members = tuple(rows), tuple(members)

    def _finish_creation(self, record):
        while True:
            try:
                with self._cv:
                    record.creation_done = True
                    self._cv.notify_all()
                return
            except BaseException:
                # The terminal outcome is durable before waiters are notified.
                continue

    def _settle_creation_error(self, record, primary):
        name = record.name
        while True:
            try:
                with self._cv:
                    committed = record.state != "creating" and record.error is None
                    if not committed:
                        record.error = primary
                        record.state = "released"
                        if self._names.get(name) is record:
                            del self._names[name]
                        for token in tuple(record.uses):
                            refs = self._runs.get(token)
                            if refs is not None:
                                refs.discard(record.ref)
                                if not refs:
                                    self._runs.pop(token, None)
                        record.uses.clear()
                break
            except BaseException:
                continue
        if committed:
            self._finish_creation(record)
            return
        try:
            self._destroy(record)
            self.mark_freed(record)
        except BaseException as cleanup_error:
            record.free_error = cleanup_error
            if self._poison is not None:
                self._poison(f"domain {name!r} creation rollback failed", cleanup_error)
        finally:
            self._finish_creation(record)

    def allocate(self, name, config, *, reuse=False):
        if not isinstance(name, str) or not name:
            raise ValueError("domain name must be a non-empty string")
        record = None
        try:
            with self._cv:
                prior = self._names.get(name)
                if prior is not None:
                    if not reuse:
                        raise ValueError(f"domain {name!r} is already live or creating")
                    difference = prior.config.difference(config)
                    if difference:
                        raise ValueError(f"domain {name!r} configuration differs: {difference}")
                    self._cv.wait_for(lambda: prior.creation_done)
                    if prior.error is not None:
                        raise prior.error
                    self.access(prior)
                    return prior.handle
                ref = CommDomainRef(self.owner_instance_id, self._next_id, 1)
                self._next_id += 1
                record = _DomainRecord(name, ref, config)
                self._records[ref] = record
                self._names[name] = record
                self._catalog(record)
                record.handle = CommDomainHandle(self, record)
            self._allocate(record)
            if record.mapping_size < config.window_size:
                raise RuntimeError("domain backend mapped less than requested window_size")
            with self._cv:
                token, _ = self._current_use() if self._current_use is not None else (None, ())
                if token is not None:
                    self._record_use(record, token)
                # A visible live record already owns its creator's run use.
                record.state = "live"
                record.creation_done = True
                self._cv.notify_all()
        except BaseException as primary:
            if record is not None:
                self._settle_creation_error(record, primary)
            raise
        return record.handle

    def import_domain(self, metadata):
        name, ref, config, catalog = decode_domain_metadata(metadata)
        with self._cv:
            prior = self._records.get(ref)
            if prior is not None:
                if prior.config != config or prior.descriptors != catalog:
                    raise ValueError("imported domain metadata conflicts with installed catalog")
                return prior
            record = _DomainRecord(name, ref, config, owner=False)
            self._catalog(record, catalog)
            record.handle = CommDomainHandle(self, record)
            self._records[ref] = record
            return record

    def admit(self, refs, token, *, inherited=()):
        with self._cv:
            # Validate the whole task/group before publishing any use.
            records = []
            for ref in refs:
                record = self._records.get(ref)
                if record is None:
                    raise KeyError(ref)
                authorized = ref in inherited
                if record.state != "live" and not (record.state == "released" and authorized):
                    raise RuntimeError(f"domain {record.name!r} is {record.state}")
                if not record.owner and not authorized:
                    raise RuntimeError("borrowed domain use requires an accepted task")
                records.append(record)
            for record in records:
                self.use(record.ref, token, inherited=record.ref in inherited)

    def get(self, key):
        with self._cv:
            record = self._names.get(key) if isinstance(key, str) else self._records.get(key)
            if record is None or record.state == "creating":
                raise KeyError(key)
            self.access(record)
            return record.handle

    def access(self, record):
        with self._cv:
            token, inherited = self._current_use() if self._current_use is not None else (None, ())
            authorized = record.ref in inherited
            if record.state != "live" and not (record.state == "released" and authorized):
                raise RuntimeError(f"domain {record.name!r} is {record.state}")
            if not record.owner and not authorized:
                raise RuntimeError("borrowed domain access requires an accepted task use")
            if token is not None:
                self.use(record.ref, token, inherited=authorized)

    def use(self, ref, token, *, inherited=False):
        with self._cv:
            record = self._records.get(ref)
            if record is None:
                raise KeyError(ref)
            if record.state != "live" and not (inherited and record.state == "released"):
                raise RuntimeError(f"domain {record.name!r} is {record.state}")
            if not record.owner and not inherited:
                raise RuntimeError("borrowed domain use requires an accepted task")
            self._record_use(record, token)

    def _record_use(self, record, token):
        record.uses.add(token)
        self._runs.setdefault(token, set()).add(record.ref)

    def release(self, record):
        with self._cv:
            if record.state in ("released", "freed"):
                return
            record.state = "released"
            if self._names.get(record.name) is record:
                del self._names[record.name]

    def finish_run(self, token, *, failed=False, settled=True):
        with self._cv:
            for ref in self._runs.pop(token, ()):
                record = self._records[ref]
                if record.owner and (failed or not record.config.retain_after_run):
                    self.release(record)
                if settled:
                    record.uses.discard(token)
                else:
                    record.unsettled.add(token)

    @property
    def cleanup_pending(self):
        with self._cv:
            return any(r.owner and r.state == "released" and not r.uses for r in self._records.values())

    def cleanup_ready(self):
        first_error = None
        with self._cv:
            candidates = [r for r in self._records.values() if r.owner and r.state == "released" and not r.uses]
        for record in candidates:
            claim = object()
            try:
                with self._cv:
                    self._cv.wait_for(lambda: record.freeing is None)
                    if record.state == "freed":
                        continue
                    if record.free_error is not None:
                        first_error = first_error or record.free_error
                        continue
                    record.freeing = claim
                self._destroy(record)
                self.mark_freed(record)
            except BaseException as exc:
                if record.freeing is not claim:
                    raise
                record.free_error = exc
                first_error = first_error or exc
            finally:
                interrupted = self._finish_cleanup(record, claim)
                first_error = first_error or interrupted
        if first_error is not None:
            raise first_error

    def _finish_cleanup(self, record, claim):
        interrupted = None
        while True:
            try:
                with self._cv:
                    if record.freeing is claim:
                        record.freeing = None
                    self._cv.notify_all()
                return interrupted
            except BaseException as exc:
                interrupted = interrupted or exc

    def mark_freed(self, record):
        """Retire a proven release; outstanding handles keep their own terminal record."""
        with self._cv:
            record.state = "freed"
            if self._records.get(record.ref) is record:
                del self._records[record.ref]

    def release_all(self):
        with self._cv:
            for record in self._records.values():
                if record.owner and record.state == "live":
                    self.release(record)

    @property
    def has_uses(self):
        with self._cv:
            return any(record.uses for record in self._records.values())

    def context_torn_down(self, *, unproven=()):
        """Retire proven mappings; local teardown cannot settle unconfirmed remote uses."""
        unproven = frozenset(unproven)
        with self._cv:
            self._names.clear()
            self._runs = {token: refs & unproven for token, refs in self._runs.items() if refs & unproven}
            for ref, record in self._records.items():
                if ref in unproven:
                    self.release(record)
                    continue
                record.uses.clear()
                record.unsettled.clear()
                record.state = "freed"
            self._records = {ref: record for ref, record in self._records.items() if ref in unproven}


def domain_window_location(descriptor):
    creator, domain_id, generation, rank, offset = _WINDOW_BODY.unpack(descriptor.body)
    return CommDomainRef(creator, domain_id, generation), rank, offset


def encode_domain_metadata(record):
    data = {
        "name": record.name,
        "ref": (record.ref.owner_instance_id.hex(), record.ref.domain_id, record.ref.generation),
        "members": [
            (m.node_worker_id, m.local_worker_id, m.global_device_rank, repr(m.target_identity))
            for m in record.config.members
        ],
        "window_size": record.config.window_size,
        "retain_after_run": record.config.retain_after_run,
        "buffers": [[getattr(spec, f.name) for f in fields(spec)] for spec in record.config.buffers],
        "catalog": [[bytes(_encode_buffer_descriptor_wire(desc)).hex() for desc in row] for row in record.descriptors],
    }
    payload = json.dumps(data, separators=(",", ":")).encode()
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("domain metadata exceeds maximum size")
    return payload


def decode_domain_metadata(payload):
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("domain metadata exceeds maximum size")
    data = json.loads(payload)
    owner, domain_id, generation = data["ref"]
    ref = CommDomainRef(bytes.fromhex(owner), domain_id, generation)
    config = _DomainConfig.capture(
        members=[_DomainMember(*m) for m in data["members"]],
        window_size=data["window_size"],
        buffers=[CommBufferSpec(*spec) for spec in data["buffers"]],
        retain_after_run=data["retain_after_run"],
        topology=(),
    )
    catalog = tuple(
        tuple(_decode_buffer_descriptor_wire(bytes.fromhex(desc)) for desc in row) for row in data["catalog"]
    )
    _validate_catalog(ref, config, catalog)
    if not isinstance(data["name"], str) or not data["name"]:
        raise ValueError("domain metadata requires a non-empty name")
    return data["name"], ref, config, catalog


class _ChipDomainStore:
    """Explicitly installed local mappings, scoped to accepted task references."""

    def __init__(self):
        self._mappings = {}
        self.task_refs = frozenset()

    def install(self, metadata, rank, allocation_id, base, size, device_ctx):
        _, ref, config, catalog = decode_domain_metadata(metadata)
        if not 0 <= rank < len(config.members) or size < config.window_size:
            raise ValueError("domain mapping rank or size does not match catalog")
        by_identity = {desc.identity: desc for desc in catalog[rank]}
        prior = self._mappings.get(ref)
        entry = (rank, allocation_id, base, size, device_ctx, by_identity)
        if prior is not None and prior != entry:
            raise ValueError("domain mapping conflicts with installed catalog")
        self._mappings[ref] = entry

    def revoke(self, allocation_id):
        self._mappings = {ref: entry for ref, entry in self._mappings.items() if entry[1] != allocation_id}

    def local_allocation(self, allocation_id):
        for ref, entry in self._mappings.items():
            if entry[1] == allocation_id:
                return ref, entry
        raise KeyError(allocation_id)

    def resolve(self, descriptor):
        ref, rank, offset = domain_window_location(descriptor)
        if ref not in self.task_refs:
            raise RuntimeError("domain tensor has no accepted task use")
        entry = self._mappings.get(ref)
        if entry is None:
            raise RuntimeError("domain mapping is not installed on this chip")
        local_rank, _, base, size, _, catalog = entry
        if rank != local_rank or catalog.get(descriptor.identity) != descriptor:
            raise ValueError("domain tensor does not match this chip's canonical catalog")
        if offset > size or descriptor.nbytes > size - offset:
            raise ValueError("domain tensor exceeds actual mapping")
        return base + offset
