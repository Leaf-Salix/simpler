# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fault boundaries for domain creation and single-slot completion publication."""

import ctypes
import multiprocessing
from contextlib import ExitStack
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import simpler.worker as worker_mod
from simpler.buffer import LocalEndpointBufferIdentityAllocator
from simpler.comm_domain import _DomainConfig, _DomainMember, _DomainRegistry
from simpler.comm_provider import (
    DeviceAllocationTarget,
    ProviderRegionStore,
    RegionAllocationContext,
    RegionEnvironmentKind,
)
from simpler.task_interface import ChipCallable


class _InterruptingCondition:
    def __init__(self, condition, *, on_enter=None, on_notify=None, on_exit=None):
        self.condition = condition
        self.on_enter = on_enter
        self.on_notify = on_notify
        self.on_exit = on_exit

    def __enter__(self):
        if self.on_enter is not None:
            callback, self.on_enter = self.on_enter, None
            callback()
        return self.condition.__enter__()

    def __exit__(self, *args):
        result = self.condition.__exit__(*args)
        if self.on_exit is not None:
            self.on_exit()
        return result

    def __getattr__(self, name):
        return getattr(self.condition, name)

    def notify_all(self):
        if self.on_notify is not None:
            callback, self.on_notify = self.on_notify, None
            callback()
        self.condition.notify_all()


def _raise(error):
    raise error


@pytest.mark.parametrize("scope", ["callback", "control"])
def test_completed_wait_inside_control_scope_defers_domain_cleanup(scope):
    from simpler.orchestrator import _callback_run  # noqa: PLC0415

    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def run():
        worker = worker_mod.Worker(level=3, num_sub_workers=0)
        registry = worker._domain_registry
        registry._allocate = lambda record: setattr(record, "mapping_size", 4096)
        frees = []
        registry._destroy = lambda record: frees.append(record.ref)
        config = _DomainConfig.capture(
            members=(_DomainMember(None, 0, 0, ("chip", 0)),),
            window_size=4096,
            buffers=(),
            retain_after_run=True,
            topology=(),
        )
        domain = registry.allocate("tp", config)
        prior = worker_mod.RunHandle(worker, 1, ())
        prior._terminal = True
        with ExitStack() as stack:
            if scope == "callback":
                stack.enter_context(worker._submit_mu.exclusive())
                stack.enter_context(_callback_run(2, worker))
            else:
                stack.enter_context(worker._control_reservation("test"))
            domain.release()
            sender.send("waiting")
            prior.wait()
            assert domain.released and not domain.freed and not frees
        worker._maintain_domains()
        assert domain.freed and frees == [domain.ref]
        worker.close()
        sender.send("freed")

    process = context.Process(target=run)
    process.start()
    sender.close()
    try:
        assert receiver.poll(5), "child did not reach wait"
        assert receiver.recv() == "waiting"
        assert receiver.poll(5), "completed wait deadlocked inside its control scope"
        assert receiver.recv() == "freed"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        receiver.close()


def test_cleanup_claim_interruption_keeps_debt_without_orphaning_waiters():
    worker = worker_mod.Worker(level=3, num_sub_workers=0)
    registry = worker._domain_registry
    registry._allocate = lambda record: setattr(record, "mapping_size", 4096)
    calls = []
    registry._destroy = lambda record: calls.append(record.ref)
    config = _DomainConfig.capture(
        members=(_DomainMember(None, 0, 0, ("chip", 0)),),
        window_size=4096,
        buffers=(),
        retain_after_run=True,
        topology=(),
    )
    domain = registry.allocate("tp", config)
    domain.release()
    interrupt = KeyboardInterrupt("cleanup claim handoff")
    fired = False

    def after_exit():
        nonlocal fired
        if domain._record.freeing and not fired:
            fired = True
            raise interrupt

    registry._cv = _InterruptingCondition(registry._cv, on_exit=after_exit)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            registry.cleanup_ready()
        assert caught.value is interrupt
        assert not domain._record.freeing
        assert domain._record.free_error is interrupt
        assert not calls and not domain.freed
        with pytest.raises(KeyboardInterrupt) as repeated:
            registry.cleanup_ready()
        assert repeated.value is interrupt
        worker._release_all_domains()
        assert not calls and domain.ref in registry.records
    finally:
        registry.context_torn_down()
        worker.close()


def test_rollback_retirement_interruption_preserves_primary_and_never_replays_free():
    worker = worker_mod.Worker(level=3, num_sub_workers=0)
    registry = worker._domain_registry
    primary, interrupt = RuntimeError("allocation failed"), KeyboardInterrupt("retirement")
    created, destroyed = [], []
    config = _DomainConfig.capture(
        members=(_DomainMember(None, 0, 0, ("chip", 0)),),
        window_size=4096,
        buffers=(),
        retain_after_run=True,
        topology=(),
    )

    def allocate(record):
        created.append(record)
        raise primary

    def destroy(record):
        destroyed.append(record.ref)
        registry._cv = _InterruptingCondition(registry._cv, on_enter=lambda: _raise(interrupt))

    registry._allocate, registry._destroy = allocate, destroy
    caught = None
    try:
        try:
            registry.allocate("tp", config)
        except BaseException as exc:
            caught = exc
        assert caught is primary
        (record,) = created
        assert record.state == "released" and record.free_error is interrupt
        assert worker._ordered_cleanup_error is not None
        with pytest.raises(KeyboardInterrupt) as repeated:
            registry.cleanup_ready()
        assert repeated.value is interrupt and destroyed == [record.ref]
    finally:
        registry.context_torn_down()
        worker.close()


@pytest.mark.parametrize("boundary", ["enter", "notify"])
def test_cleanup_terminal_publication_interrupt_does_not_strand_contender(boundary):
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def run():
        registry = _DomainRegistry(
            owner_instance_id=b"creator1",
            mint_identity=lambda: None,
            allocate=lambda record: setattr(record, "mapping_size", 4096),
            destroy=lambda _record: None,
        )
        config = _DomainConfig.capture(
            members=(_DomainMember(None, 0, 0, ("chip", 0)),),
            window_size=4096,
            buffers=(),
            retain_after_run=True,
            topology=(),
        )
        domain = registry.allocate("tp", config)
        domain.release()
        entered, parked, finish = Event(), Event(), Event()
        interrupt = KeyboardInterrupt("cleanup terminal publication")
        calls, errors = [], []

        class Condition(_InterruptingCondition):
            def wait_for(self, predicate):
                if not predicate():
                    parked.set()
                return self.condition.wait_for(predicate)

        registry._cv = Condition(registry._cv)

        def destroy(record):
            calls.append(record.ref)
            entered.set()
            assert finish.wait(5)

        retire = registry.mark_freed

        def mark_freed(record):
            retire(record)

            def callback():
                raise interrupt

            if boundary == "enter":
                registry._cv.on_enter = callback
            else:
                registry._cv.on_notify = callback

        registry._destroy, registry.mark_freed = destroy, mark_freed

        def clean():
            try:
                registry.cleanup_ready()
            except BaseException as exc:
                errors.append(exc)

        owner, contender = Thread(target=clean, daemon=True), Thread(target=clean, daemon=True)
        owner.start()
        assert entered.wait(5)
        contender.start()
        assert parked.wait(5)
        sender.send("parked")
        finish.set()
        owner.join(5)
        contender.join(1)
        assert not owner.is_alive() and not contender.is_alive()
        assert calls == [domain.ref] and errors == [interrupt]
        assert domain.freed and not domain._record.freeing and registry.records == {}
        sender.send("finished")

    process = context.Process(target=run)
    process.start()
    sender.close()
    try:
        assert receiver.poll(5) and receiver.recv() == "parked"
        assert receiver.poll(5), "cleanup contender was stranded"
        assert receiver.recv() == "finished"
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        receiver.close()


@pytest.mark.parametrize("debt", ["free", "unsettled"])
def test_local_teardown_does_not_prove_failed_remote_domain_release(monkeypatch, debt):
    worker = worker_mod.Worker(level=4, num_sub_workers=0)
    worker._remote_worker_ids = [7]
    session = worker_mod._RemoteSession(7, 9, "127.0.0.1", 1, "127.0.0.1", 2, 99)
    worker._remote_sessions = [session]
    registry = worker._domain_registry
    registry._allocate = lambda record: setattr(record, "mapping_size", 4096)
    config = _DomainConfig.capture(
        members=(_DomainMember(7, 0, 0, ("remote", 7)),),
        window_size=4096,
        buffers=(),
        retain_after_run=True,
        topology=(),
    )
    domain = registry.allocate("remote", config)
    domain._record.backend = {"kind": "global", "nodes": [7]}
    if debt == "free":
        domain._record.free_error = RuntimeError("remote free failed")
    else:
        registry.use(domain.ref, "unknown-run")
        registry.finish_run("unknown-run", settled=False)
    domain.release()
    shutdown = []
    monkeypatch.setattr(worker, "_close_remote_session", lambda value: shutdown.append(value))

    def local_teardown():
        worker._release_all_domains()
        worker._release_remote_sessions()

    monkeypatch.setattr(worker, "_teardown_ready_tree", local_teardown)
    try:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            worker.close()
        assert shutdown == [session]
        assert domain.released and not domain.freed and domain.ref in registry.records
        if debt == "free":
            assert domain._record.free_error is not None
        else:
            assert domain._record.uses == domain._record.unsettled == {"unknown-run"}
        assert worker._remote_sessions == [session]
    finally:
        registry.context_torn_down()
        worker._remote_sessions.clear()
        worker._close_completion = None
        worker.close()


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit", "after_release"])
def test_creation_publication_interrupt_always_settles_its_record(boundary):
    interrupt = KeyboardInterrupt("creation publication")
    destroyed = []
    created = []
    config = _DomainConfig.capture(
        members=(_DomainMember(None, 0, 0, ("chip", 0)),),
        window_size=4096,
        buffers=(),
        retain_after_run=True,
        topology=(),
    )
    registry = _DomainRegistry(
        owner_instance_id=b"creator1",
        mint_identity=lambda: None,
        allocate=lambda _: None,
        destroy=lambda record: destroyed.append(record.ref),
        current_use=lambda: ("creator-run", ()),
    )

    def allocate(record):
        created.append(record)
        record.mapping_size = 4096

        def interrupt_notification():
            if boundary == "after_release":
                registry.release(record)
            raise interrupt

        registry._cv = _InterruptingCondition(
            registry._cv,
            on_enter=(lambda: _raise(interrupt)) if boundary == "before_commit" else None,
            on_notify=interrupt_notification if boundary != "before_commit" else None,
        )

    registry._allocate = allocate
    with pytest.raises(KeyboardInterrupt) as caught:
        registry.allocate("tp", config)
    assert caught.value is interrupt
    (record,) = created
    assert record.creation_done
    if boundary == "before_commit":
        assert record.state == "freed"
        assert destroyed == [record.ref]
        assert not record.uses
        assert registry.live_domains() == {}
        assert registry.records == {}
    else:
        assert record.state == ("live" if boundary == "after_commit" else "released")
        assert not destroyed
        assert record.uses == {"creator-run"}
        registry.finish_run("creator-run", failed=True)
        registry.cleanup_ready()
        assert record.state == "freed" and destroyed == [record.ref]


@pytest.mark.parametrize("settled", [False, True])
def test_serial_chip_failure_publishes_the_native_domain_use_proof(monkeypatch, settled):
    def fail(*_args, **_kwargs):
        raise RuntimeError("terminal kernel failure")

    native_run = SimpleNamespace(wait=fail, domain_use_settled=settled)
    chip = SimpleNamespace(run_materialized=fail, _submit_chip_run_materialized=lambda *_a, **_k: native_run)
    owner = b"creator1"
    store = ProviderRegionStore(
        RegionAllocationContext(environment_kind=RegionEnvironmentKind.SIM, target=DeviceAllocationTarget(0)),
        identity_allocator=LocalEndpointBufferIdentityAllocator(owner),
    )
    target = ChipCallable.build(signature=[], func_name="x", binary=b"0", children=[])
    result = []
    with memoryview(bytearray(worker_mod.mailbox_size(1))) as mailbox:
        with memoryview(bytearray(worker_mod.MAILBOX_RUN_ENTRY_STRIDE)) as entry:

            def serve(_buf, _state, *, handle_task, **_kwargs):
                result.append(handle_task(mailbox, entry))

            monkeypatch.setattr(worker_mod, "_run_mailbox_loop", serve)
            address = ctypes.addressof(ctypes.c_char.from_buffer(mailbox))
            worker_mod._run_chip_main_loop(
                chip,
                mailbox,
                address,
                address + worker_mod._OFF_STATE,
                0,
                {0: target},
                {bytes(32): 0},
                {},
                owner,
                prepared={0},
                granted_sets=1,
                chip_platform="a2a3sim",
                provider_region_store=store,
            )
    code, message, proof = result[0]
    assert code == 1 and "terminal kernel failure" in message
    assert proof is settled
