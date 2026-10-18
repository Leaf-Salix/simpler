# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified communication-domain contracts, independent of the allocation backend."""

from dataclasses import replace
from itertools import count
from threading import Event, Thread

import pytest
from simpler.buffer import CanonicalIdentity, ImportContext, ImportRegistry
from simpler.comm_domain import (
    CommBufferSpec,
    _ChipDomainStore,
    _DomainConfig,
    _DomainMember,
    _DomainRegistry,
    encode_domain_metadata,
)
from simpler.comm_endpoints import DEVICE_AICPU
from simpler.task_interface import DataType


@pytest.fixture
def domains():
    ids = count(1)
    allocations, frees = [], []

    def allocate(record):
        allocations.append(record.ref)
        record.mapping_size = 8192

    registry = _DomainRegistry(
        owner_instance_id=b"creator1",
        mint_identity=lambda: CanonicalIdentity(b"creator1", next(ids), 1),
        allocate=allocate,
        destroy=lambda record: frees.append(record.ref),
    )
    return registry, allocations, frees


def config(**changes):
    values = {
        "members": (_DomainMember(None, 2, 2, ("chip", 2)), _DomainMember(None, 0, 0, ("chip", 0))),
        "window_size": 4096,
        "buffers": [CommBufferSpec("scratch", "float32", 16, 64)],
        "retain_after_run": True,
        "topology": ("sim", "session-1"),
    }
    values.update(changes)
    return _DomainConfig.capture(**values)


def test_same_configuration_reuses_the_owner_handle_and_catalog(domains):
    registry, allocations, _ = domains
    first = registry.allocate("tp", config(), reuse=True)
    tensor = first.member(0).buffers["scratch"].tensor([16], DataType.FLOAT32)
    second = registry.allocate("tp", config(), reuse=True)
    assert second is first
    assert registry.get("tp") is first
    assert registry.get(first.ref) is first
    assert second.member(0).buffers["scratch"].identity == tensor.buffer.identity
    assert allocations == [first.ref]
    assert first.requested_window_size == 4096
    assert first.mapping_size == 8192
    assert first.member(0).local_worker_id == 2
    assert first.member(1).local_worker_id == 0


@pytest.mark.parametrize("field", ("dtype", "count", "nbytes", "load_from_host", "store_to_host"))
def test_every_buffer_definition_field_participates_in_reuse(domains, field):
    registry, allocations, _ = domains
    registry.allocate("tp", config(), reuse=True)
    different = {"dtype": "uint32", "count": 8, "nbytes": 32, "load_from_host": True, "store_to_host": True}
    spec = replace(CommBufferSpec("scratch", "float32", 16, 64), **{field: different[field]})
    with pytest.raises(ValueError, match=field):
        registry.allocate("tp", config(buffers=[spec]), reuse=True)
    assert len(allocations) == 1


def test_default_retain_value_and_ordered_members_are_not_ignored(domains):
    registry, allocations, _ = domains
    initial = config()
    registry.allocate("tp", initial, reuse=True)
    for changed, field in (
        (replace(initial, retain_after_run=False), "retain_after_run"),
        (replace(initial, members=initial.members[::-1]), "members"),
        (replace(initial, window_size=2048), "window_size"),
        (replace(initial, topology=("sim", "session-2")), "topology"),
    ):
        with pytest.raises(ValueError, match=field):
            registry.allocate("tp", changed, reuse=True)
    assert len(allocations) == 1


def test_configuration_is_a_value_snapshot(domains):
    registry, _, _ = domains
    spec = CommBufferSpec("scratch", "float32", 16, 64)
    original = [spec]
    snapshot = config(buffers=original)
    handle = registry.allocate("tp", snapshot, reuse=True)
    spec.nbytes = 128
    original.clear()
    assert registry.allocate("tp", config(), reuse=True) is handle
    assert handle.buffers[0].nbytes == 64


def test_lookup_and_duplicate_allocate_do_not_create(domains):
    registry, allocations, _ = domains
    with pytest.raises(KeyError):
        registry.get("missing")
    first = registry.allocate("tp", config(), reuse=False)
    with pytest.raises(ValueError, match="already"):
        registry.allocate("tp", config(), reuse=False)
    assert allocations == [first.ref]


def test_successful_cleanup_retires_records_and_rejects_old_refs(domains):
    registry, _, frees = domains
    old_refs = []
    for _ in range(100):
        handle = registry.allocate("tp", config())
        old_refs.append(handle.ref)
        handle.release()
        registry.cleanup_ready()
        assert handle.freed
        assert registry.records == {}
        with pytest.raises(KeyError):
            registry.get(handle.ref)
        with pytest.raises(RuntimeError, match="freed"):
            handle.member(0)
    assert len(set(old_refs)) == 100
    assert frees == old_refs


def test_successful_creation_rollback_retires_its_record(domains):
    registry, _, frees = domains
    captured = []

    def fail(record):
        captured.append(record)
        raise RuntimeError("allocation failed")

    registry._allocate = fail
    with pytest.raises(RuntimeError, match="allocation failed"):
        registry.allocate("tp", config())
    assert captured[0].state == "freed"
    assert frees == [captured[0].ref]
    assert registry.records == {}


def test_context_teardown_retires_records_but_keeps_handle_state(domains):
    registry, _, _ = domains
    handle = registry.allocate("tp", config())
    registry.use(handle.ref, "unsettled")
    registry.finish_run("unsettled", settled=False)
    registry.context_torn_down()
    assert handle.freed
    assert not registry.has_uses and registry.records == {}


def test_domain_materialization_does_not_retain_revoked_catalogs(domains):
    registry, _, _ = domains
    store = _ChipDomainStore()
    imports = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, domain_resolver=store.resolve))
    for allocation in range(10):
        handle = registry.allocate("tp", config())
        store.install(encode_domain_metadata(handle._record), 0, allocation, 0x10000, 8192, 0x20000)
        store.task_refs = frozenset([handle.ref])
        descriptor = handle.member(0).buffers["scratch"].to_descriptor()
        assert imports.materialize(descriptor).base == 0x10000
        store.revoke(allocation)
        handle.release()
        registry.cleanup_ready()
        with pytest.raises(RuntimeError, match="not installed"):
            imports.materialize(descriptor)
    assert imports._by_identity == {}


def test_domain_forwarding_does_not_retain_retired_catalogs(domains):
    from simpler.worker import Worker  # noqa: PLC0415

    registry, _, _ = domains
    worker = Worker(level=3, num_sub_workers=0)
    try:
        for _ in range(10):
            domain = registry.allocate("tp", config())
            descriptor = domain.member(0).buffers["scratch"].to_descriptor()
            forwarded = worker._reexport(descriptor)
            assert forwarded.to_descriptor() == descriptor
            domain.release()
            registry.cleanup_ready()
        assert worker._reexport_by_source == {}
    finally:
        worker.close()


def test_release_waits_for_every_using_run_and_allows_same_name_rebuild(domains):
    registry, _, frees = domains
    first = registry.allocate("tp", config(), reuse=True)
    registry.use(first.ref, "run-1")
    registry.use(first.ref, "run-2")
    first.release()
    first.release()
    with pytest.raises(KeyError):
        registry.get("tp")
    with pytest.raises(RuntimeError, match="released"):
        first.member(0)
    with pytest.raises(RuntimeError, match="released"):
        registry.use(first.ref, "new-run")
    second = registry.allocate("tp", config(), reuse=True)
    assert first.ref != second.ref
    assert first.released and not first.freed
    registry.finish_run("run-1")
    registry.cleanup_ready()
    assert not frees
    registry.finish_run("run-2")
    registry.cleanup_ready()
    assert frees == [first.ref]
    assert first.freed
    assert registry.get("tp") is second


def test_failed_later_run_closes_retained_domain_but_keeps_other_uses(domains):
    registry, _, frees = domains
    domain = registry.allocate("tp", config(), reuse=True)
    registry.use(domain.ref, "first")
    registry.finish_run("first")
    assert not domain.released
    registry.use(domain.ref, "second")
    registry.use(domain.ref, "third")
    registry.finish_run("second", failed=True)
    registry.cleanup_ready()
    assert domain.released and not domain.freed and not frees
    registry.finish_run("third")
    registry.cleanup_ready()
    assert domain.freed and frees == [domain.ref]


def test_unproven_completion_preserves_the_use_and_blocks_free(domains):
    registry, _, frees = domains
    domain = registry.allocate("tp", config(), reuse=True)
    registry.use(domain.ref, "child-unsettled")
    registry.finish_run("child-unsettled", failed=True, settled=False)
    registry.cleanup_ready()
    assert domain.released and not domain.freed
    assert not frees
    assert registry.unsettled_refs == {domain.ref}


def test_backend_release_failure_remains_tracked_and_is_not_replayed(domains):
    registry, _, _ = domains
    attempts = []

    def fail(record):
        attempts.append(record.ref)
        raise RuntimeError("partial free")

    registry._destroy = fail
    domain = registry.allocate("tp", config(), reuse=True)
    domain.release()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="partial free"):
            registry.cleanup_ready()
    assert attempts == [domain.ref]
    assert not domain.freed
    assert domain.ref in registry.records


def test_same_configuration_waits_for_one_creation_without_publishing_it(domains):
    registry, allocations, _ = domains
    entered, proceed = Event(), Event()
    original = registry._allocate

    def allocate(record):
        entered.set()
        assert proceed.wait(5)
        original(record)

    registry._allocate = allocate
    handles, errors = [], []

    def create():
        try:
            handles.append(registry.allocate("tp", config(), reuse=True))
        except BaseException as exc:
            errors.append(exc)

    first = Thread(target=create)
    second = Thread(target=create)
    first.start()
    assert entered.wait(5)
    try:
        with pytest.raises(KeyError):
            registry.get("tp")
        with pytest.raises(ValueError, match="window_size"):
            registry.allocate("tp", config(window_size=2048), reuse=True)
        second.start()
    finally:
        proceed.set()
        first.join(5)
        second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors and len(handles) == 2 and handles[0] is handles[1]
    assert len(allocations) == 1


def test_borrowed_handle_requires_each_tasks_inherited_use_and_cannot_release(domains):
    owner, _, _ = domains
    domain = owner.allocate("tp", config(), reuse=True)
    scope = [None, frozenset()]
    borrower = _DomainRegistry(
        owner_instance_id=b"borrower",
        mint_identity=lambda: None,
        allocate=lambda _: None,
        destroy=lambda _: pytest.fail("borrowed domain freed locally"),
        current_use=lambda: tuple(scope),
    )
    record = borrower.import_domain(encode_domain_metadata(domain._record))
    record.state = "live"
    record.mapping_size = domain.mapping_size
    with pytest.raises(RuntimeError, match="accepted task"):
        borrower.get(domain.ref)
    scope[:] = ["accepted-1", frozenset([domain.ref])]
    borrowed = borrower.get(domain.ref)
    assert borrowed.borrowed
    assert borrowed.member(0).buffers["scratch"].to_descriptor() == domain.member(0).buffers["scratch"].to_descriptor()
    with pytest.raises(RuntimeError, match="borrowed"):
        borrowed.release()
    with pytest.raises(RuntimeError, match="borrowed"):
        borrowed.__enter__()
    borrower.finish_run("accepted-1")
    scope[:] = ["ordinary-next-run", frozenset()]
    with pytest.raises(RuntimeError, match="accepted task"):
        borrowed.member(0)


def test_chip_mapping_requires_exact_rank_catalog_and_rechecks_cached_use(domains):
    registry, _, _ = domains
    domain = registry.allocate("tp", config(), reuse=True)
    store = _ChipDomainStore()
    store.install(encode_domain_metadata(domain._record), 0, 987, 0x10000, 8192, 0x20000)
    cache = ImportRegistry(ImportContext(deployment=DEVICE_AICPU, domain_resolver=store.resolve))
    descriptor = domain.member(0).buffers["scratch"].to_descriptor()
    with pytest.raises(RuntimeError, match="accepted task"):
        cache.materialize(descriptor)
    store.task_refs = frozenset([domain.ref])
    assert cache.materialize(descriptor).base == 0x10000
    with pytest.raises(ValueError, match="canonical catalog"):
        cache.materialize(domain.member(1).buffers["scratch"].to_descriptor())
    store.task_refs = frozenset()
    with pytest.raises(RuntimeError, match="accepted task"):
        cache.materialize(descriptor)
    store.task_refs = frozenset([domain.ref])
    store.revoke(987)
    with pytest.raises(RuntimeError, match="not installed"):
        cache.materialize(descriptor)


def test_group_use_admission_is_atomic(domains):
    registry, _, _ = domains
    first = registry.allocate("a", config(), reuse=True)
    second = registry.allocate("b", config(), reuse=True)
    second.release()
    with pytest.raises(RuntimeError, match="released"):
        registry.admit([first.ref, second.ref], "group")
    assert not first._record.uses and not second._record.uses


@pytest.mark.parametrize("failure", [RuntimeError("free failed"), KeyboardInterrupt("free interrupted")])
def test_cleanup_failure_keeps_claim_and_continues_other_domains(domains, failure):
    registry, _, _ = domains
    first = registry.allocate("first", config())
    second = registry.allocate("second", config())
    attempted = []

    def destroy(record):
        attempted.append(record.ref)
        if record.ref == first.ref:
            raise failure

    registry._destroy = destroy
    first.release()
    second.release()
    for _ in range(2):
        with pytest.raises(type(failure)) as caught:
            registry.cleanup_ready()
        assert caught.value is failure
    assert attempted == [first.ref, second.ref]
    assert second.freed and not first.freed


def test_concurrent_cleanup_waits_for_one_backend_outcome(domains):
    registry, _, _ = domains
    domain = registry.allocate("tp", config())
    domain.release()
    entered, finish, second_entered = Event(), Event(), Event()
    calls, errors = [], []

    def destroy(record):
        calls.append(record.ref)
        entered.set()
        assert finish.wait(5)

    registry._destroy = destroy

    def clean(second=False):
        if second:
            second_entered.set()
        try:
            registry.cleanup_ready()
        except BaseException as exc:
            errors.append(exc)

    owner, contender = Thread(target=clean), Thread(target=clean, args=(True,))
    owner.start()
    assert entered.wait(5)
    contender.start()
    try:
        assert second_entered.wait(5)
        assert not domain.freed
    finally:
        finish.set()
        owner.join(5)
        contender.join(5)
    assert not owner.is_alive() and not contender.is_alive()
    assert not errors and calls == [domain.ref] and domain.freed


def test_worker_maintenance_waits_for_unrelated_accepted_run(domains):
    from simpler.worker import RunHandle, Worker, _RunResources  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    registry, _, frees = domains
    worker._comm_domains = registry
    domain = registry.allocate("tp", config())
    domain.release()
    entered, finish, cleaned = Event(), Event(), Event()
    resources = _RunResources()
    handle = RunHandle(worker, 2, (), resources)
    worker._accepted_run_handles.add(handle)

    def wait_for_serialization():
        entered.set()
        assert finish.wait(5)
        worker._accepted_run_handles.discard(handle)

    handle._wait_for_serialization = wait_for_serialization
    errors = []

    def maintain():
        try:
            worker._maintain_domains()
            cleaned.set()
        except BaseException as exc:
            errors.append(exc)

    maintainer = Thread(target=maintain)
    maintainer.start()
    try:
        assert entered.wait(5)
        assert not frees and not cleaned.is_set()
    finally:
        finish.set()
        maintainer.join(5)
    assert not maintainer.is_alive() and not errors
    assert domain.freed and frees == [domain.ref]


def test_timed_wait_leaves_pending_cleanup_for_next_safe_boundary(domains):
    from simpler.worker import RunHandle, Worker  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    registry, _, frees = domains
    worker._comm_domains = registry
    domain = registry.allocate("tp", config())
    domain.release()
    handle = RunHandle(worker, 1, ())
    handle._terminal = True
    worker._maintain_domains = lambda: pytest.fail("finite wait entered unbounded cleanup")
    handle.wait(timeout=0.01)
    assert registry.cleanup_pending and not frees


def test_unknown_worker_completion_keeps_domain_until_context_teardown():
    from types import SimpleNamespace  # noqa: PLC0415

    from simpler.worker import RunHandle, Worker, _RunResources  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    registry = worker._domain_registry
    registry._allocate = lambda record: setattr(record, "mapping_size", 8192)
    frees = []
    registry._destroy = lambda record: frees.append(record.ref)
    domain = registry.allocate("tp", config())
    resources = _RunResources()
    registry.use(domain.ref, resources.domain_use_token)
    worker._orch = SimpleNamespace(_release_run=lambda _: None)
    handle = RunHandle(worker, 1, (), resources)
    worker._accepted_run_handles.add(handle)
    result = worker._finalize_run_handle(handle, 1, None)
    assert isinstance(result, RuntimeError)
    assert "unsettled" in str(result)
    worker._release_all_domains()
    assert domain.released and not domain.freed and not frees
    assert resources.domain_use_token in domain._record.uses
    registry.context_torn_down()
    assert domain.freed and not registry.has_uses


@pytest.mark.parametrize("part", ["backend", "access", "rank", "extent", "duplicate"])
def test_chip_install_rejects_noncanonical_catalog(domains, part):
    import json  # noqa: PLC0415

    from _task_interface import (  # noqa: PLC0415
        AccessMode,
        AddressSpace,
        BackendKind,
        BufferDescriptor,
        _encode_buffer_descriptor_wire,
    )

    registry, _, _ = domains
    domain = registry.allocate("tp", config())
    data = json.loads(encode_domain_metadata(domain._record))
    original = domain._record.descriptors[0][0]
    values = dict(
        identity=original.identity,
        address_space=AddressSpace.DEVICE,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.DOMAIN_WINDOW,
        nbytes=original.nbytes,
        body=original.body,
    )
    if part == "backend":
        values.update(backend_kind=BackendKind.VMM_WINDOW, body=b"\1" + b"\0" * 7)
    elif part == "access":
        values["access"] = AccessMode.READ
    elif part == "rank":
        values["body"] = domain._record.descriptors[1][0].body
    elif part == "extent":
        values["nbytes"] = 32
    elif part == "duplicate":
        other = domain._record.descriptors[1][0]
        values.update(body=other.body)
        data["catalog"][1][0] = bytes(_encode_buffer_descriptor_wire(BufferDescriptor(**values))).hex()
    if part != "duplicate":
        data["catalog"][0][0] = bytes(_encode_buffer_descriptor_wire(BufferDescriptor(**values))).hex()
    with pytest.raises(ValueError, match="catalog"):
        _ChipDomainStore().install(json.dumps(data).encode(), 0, 1, 0x1000, 8192, 1)


def test_local_blob_rejects_domain_tensor_without_declared_ref(domains):
    import ctypes  # noqa: PLC0415

    from _task_interface import read_args_from_blob  # noqa: PLC0415

    from tests.ut.py.test_worker._wire_blob import encode_blob  # noqa: PLC0415

    registry, _, _ = domains
    domain = registry.allocate("tp", config())
    tensor = domain.member(0).buffers["scratch"].tensor([16], DataType.FLOAT32)
    blob = ctypes.create_string_buffer(encode_blob([tensor]))
    with pytest.raises(ValueError, match="domain reference"):
        read_args_from_blob(ctypes.addressof(blob), len(blob))


def test_later_run_cannot_overwrite_an_unsettled_descendant_proof():
    from types import SimpleNamespace  # noqa: PLC0415

    from simpler.worker import RunHandle, Worker, _RunResources  # noqa: PLC0415

    worker = Worker(level=3, num_sub_workers=0)
    registry = worker._domain_registry
    registry._allocate = lambda record: setattr(record, "mapping_size", 8192)
    registry._destroy = lambda record: pytest.fail("unsettled device work reached physical free")
    domain = registry.allocate("used-by-first", config())
    pending = registry.allocate("unrelated-pending", config())
    resources = _RunResources()
    registry.use(domain.ref, resources.domain_use_token)
    first = RunHandle(worker, 1, (), resources)
    second = RunHandle(worker, 2, ())
    worker._accepted_run_handles.update((first, second))
    worker._wait_run_handle = lambda *_: True
    worker._orch = SimpleNamespace(_release_run=lambda _: None, _run_domain_use_settled=lambda run: run != 1)
    pending.release()
    with pytest.raises(RuntimeError, match="unsettled"):
        first.wait()
    assert second.done
    assert not worker._domain_uses_settled()
    assert not domain.freed and not pending.freed
