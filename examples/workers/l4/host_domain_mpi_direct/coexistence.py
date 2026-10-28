# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate independent Host and NPU domain lifetimes on two MPI nodes."""

import json

from simpler.comm_domain import CommDomainRef
from simpler.remote_l3_session import get_inner_handle
from simpler.task_interface import CommBufferSpec, DataType, HostWindowSpec, TaskArgs, TensorArgType
from simpler.worker import RemoteCallable

from examples.workers.l4.host_domain_mpi_direct.main import (
    COUNT,
    HOST_WINDOW,
    RESULT,
    SOURCE,
    SPAN,
    TARGET,
    WINDOW,
    _build_kernel,
    _pattern,
)

MODULE = "examples.workers.l4.host_domain_mpi_direct.coexistence"
_KEEPALIVE = []


def _copy_peer(orch, args, cfg):
    if not __debug__:
        raise RuntimeError("coexistence checks require Python without -O")
    domain = orch.get_domain(
        CommDomainRef(int(args.scalar(8)).to_bytes(8, "little"), int(args.scalar(0)), int(args.scalar(9)))
    )
    node, seed, phase = (int(args.scalar(i)) for i in range(1, 4))
    if seed:
        part = domain.local_host()
        expected = _pattern(seed, HOST_WINDOW)
        if phase:
            previous = _pattern(int(args.scalar(10)) + node, HOST_WINDOW)
            actual = bytearray(HOST_WINDOW)
            part.read(0, actual)
            assert actual == previous
        if phase != 2:
            part.write(0, expected)
        actual = bytearray(HOST_WINDOW)
        part.read(0, actual)
        assert actual == expected
        if phase == 2:
            return
        for chip in (0, 1):
            orch.copy_domain(
                domain.member(2 * node + chip).buffers["payload"],
                domain.host(node),
                worker=chip,
                nbytes=SPAN,
                dst_offset=RESULT,
                src_offset=SOURCE,
            )
    if phase == 0:
        return
    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little") for i in range(4, 8))
    _KEEPALIVE.clear()
    for chip in (0, 1):
        rank = 2 * node + chip
        context = domain.local_context(rank)
        payload = context.buffers["payload"]
        task = TaskArgs()
        task.add_tensor(payload.tensor([COUNT], DataType.FLOAT32, byte_offset=SOURCE), TensorArgType.INPUT)
        task.add_tensor(payload.tensor([COUNT], DataType.FLOAT32, byte_offset=TARGET), TensorArgType.OUTPUT_EXISTING)
        for value in (context.device_ctx, rank ^ 2, -1):
            task.add_scalar(value)
        _KEEPALIVE.append(task)
        orch.submit_next_level(get_inner_handle(digest.hex()), task, cfg, worker=chip)


def controller(context):
    executors = context.topology.executors
    if len(executors) != 2 or len({spec.host for spec in executors}) != 2:
        raise ValueError("coexistence requires two distinct executor hosts")
    if any(len(spec.device_ids) != 2 for spec in executors):
        raise ValueError("coexistence requires two chips per host")
    if tuple(spec.worker_id for spec in executors) != (0, 1):
        raise ValueError("coexistence requires executor worker ids 0 and 1")
    if not __debug__:
        raise RuntimeError("coexistence checks require Python without -O")
    worker = context.create_worker(num_sub_workers=0)
    kernel = worker.register(_build_kernel())
    route = worker.register(RemoteCallable(MODULE + ":_copy_peer"), workers=[0, 1])
    worker.init()
    for order in (("host", "ep", "tp"), ("ep", "tp", "host")):
        run_coexistence(worker, kernel, route, order)
    print(json.dumps({"result": "COEXISTENCE_PASS", "nodes": 2, "chips": 4, "release_orders": 2}), flush=True)


def run_coexistence(worker, kernel, route, release_order):
    domains, expected = {}, {}
    keepalive = []

    def allocate(orch, _args, _cfg):
        for index, name in enumerate(("ep", "host", "tp"), 1):
            domain = orch.get_or_allocate_domain(
                name=name,
                members=[(node, chip) for node in range(2) for chip in (0, 1)],
                window_size=WINDOW,
                buffers=[CommBufferSpec("payload", "float32", WINDOW // 4, WINDOW)],
                host_windows={node: HostWindowSpec(0, HOST_WINDOW) for node in range(2)} if name == "host" else None,
                retain_after_run=True,
            )
            domains[name] = domain
            assert len(domain.members) == 4
            for rank in range(4):
                expected[name, rank] = bytearray(_pattern(10 * index + rank, WINDOW))
                orch.copy_to_domain(domain, rank, expected[name, rank], buffer="payload")
        assert len({domain.domain_id for domain in domains.values()}) == 3
        assert len({m.buffers["payload"].identity for d in domains.values() for m in d.members}) == 12

    def verify(orch, _args, _cfg):
        for name, domain in domains.items():
            assert orch.get_domain(name) is domain
            for rank in range(4):
                assert orch.copy_from_domain(domain, rank, WINDOW, buffer="payload") == expected[name, rank]

    host_seed, serial = 0, 0

    def dispatch(name, phase, seed=0, previous=0):
        domain = domains[name]

        def submit(orch, _args, cfg):
            orch.get_domain(domain.ref)
            keepalive.clear()
            for node in range(2):
                args = TaskArgs()
                for value in (domain.domain_id, node, seed + node if seed else 0, phase):
                    args.add_scalar(value)
                for offset in range(0, 32, 8):
                    args.add_scalar(int.from_bytes(kernel.digest[offset : offset + 8], "little"))
                args.add_scalar(int.from_bytes(domain.ref.owner_instance_id, "little"))
                args.add_scalar(domain.generation)
                args.add_scalar(previous)
                keepalive.append(args)
                orch.submit_next_level(route, args, cfg, worker=node)

        worker.run(submit)

    def check():
        worker.run(verify)
        if host_seed and "host" in domains:
            dispatch("host", 2, host_seed, host_seed)

    def exercise(name, initialize=False):
        nonlocal host_seed, serial
        serial += 1

        def update_source(orch, _args, _cfg):
            for rank in range(4):
                data = _pattern(100 + 4 * serial + rank, SPAN)
                orch.copy_to_domain(domains[name], rank, data, buffer="payload", offset=SOURCE)
                expected[name, rank][SOURCE : SOURCE + SPAN] = data

        worker.run(update_source)
        seed = 80 + 2 * serial if name == "host" else 0
        dispatch(name, int(not initialize), seed, host_seed)
        if seed:
            host_seed = seed
        for rank in range(4):
            if seed:
                expected[name, rank][RESULT : RESULT + SPAN] = _pattern(seed + rank // 2, HOST_WINDOW)[
                    SOURCE : SOURCE + SPAN
                ]
            if not initialize:
                expected[name, rank][TARGET : TARGET + SPAN] = expected[name, rank ^ 2][SOURCE : SOURCE + SPAN]
        check()

    worker.run(allocate)
    worker.run(verify)
    exercise("host", initialize=True)
    for retiring in release_order:
        for name in domains:
            exercise(name)
        released = domains.pop(retiring)
        released.release()
        assert released.released and released.freed
        assert set(worker.live_domains) == set(domains)
        check()
    assert not worker.live_domains
