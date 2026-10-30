#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate CPU and kernel access to one DRAM and two HBM windows per MPI executor."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

from simpler.buffer import AccessMode, AddressSpace
from simpler.comm_domain import CommDomainRef
from simpler.mpi_direct_supervisor import run_supervisor
from simpler.remote_l3_session import get_inner_handle
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    HostWindowSpec,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
)
from simpler.worker import RemoteCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from tests.st.worker.comm_domain.test_host_sparse_sdma import SIZE, N, build_sparse_sdma

MODULE = "tests.st.worker.comm_domain.host_domain_matrix"
WINDOW = 4096
HOST_WINDOW = 16384
COUNT = 256
SPAN = COUNT * 4
TARGET = 256
SOURCE = 1536
RESULT = 2816
_EXPECTED: dict[int, bytearray] = {}
_KEEPALIVE: list[TaskArgs] = []


def _pattern(seed: int, nbytes: int) -> bytes:
    return struct.pack(f"<{nbytes // 4}f", *(float(seed * 1000 + i % 251) for i in range(nbytes // 4)))


def _pools(domain):
    nodes = sorted({member.node_worker_id for member in domain.members})
    return [domain.host(node) for node in nodes] + [member.buffers["payload"] for member in domain.members]


def _reset(orch, domain, node, nodes):
    _EXPECTED.clear()
    for index in range(3 * nodes):
        _EXPECTED[index] = bytearray(_pattern(index + 1, HOST_WINDOW if index < nodes else WINDOW))
    orch.copy_to_host_domain(domain, _EXPECTED[node], node=node)
    for rank, member in enumerate(domain.members):
        if member.node_worker_id == node:
            data = _EXPECTED[nodes + rank]
            orch.copy_to_domain(domain, rank, data, buffer="payload")


def _event_expected(actor, target, write, serial, nodes):
    node = actor if actor < nodes else (actor - nodes) // 2
    local = _EXPECTED[node]
    if write:
        source = _pattern(100 + serial, SPAN)
        local[SOURCE : SOURCE + SPAN] = source
        _EXPECTED[target][TARGET : TARGET + SPAN] = source
    else:
        local[RESULT : RESULT + SPAN] = _EXPECTED[target][TARGET : TARGET + SPAN]


def _verify(orch, domain, node, nodes):
    actual = orch.copy_from_host_domain(domain, HOST_WINDOW, node=node)
    if actual != _EXPECTED[node]:
        raise AssertionError(f"Host full-window/guard mismatch node={node}")
    for rank, member in enumerate(domain.members):
        if member.node_worker_id == node:
            observed = orch.copy_from_domain(domain, rank, WINDOW, buffer="payload")
            if observed != _EXPECTED[nodes + rank]:
                raise AssertionError(f"HBM full-window/guard mismatch rank={rank}")
    actual = orch.copy_from_host_domain(domain, HOST_WINDOW, node=node)
    if actual != _EXPECTED[node]:
        raise AssertionError(f"Host changed during HBM verification node={node}")


def node_orch(orch, args, cfg, chip_handle=None):
    phase, node, nodes, actor, target, write, serial = (int(args.scalar(i)) for i in range(7))
    domain = orch.get_domain(
        CommDomainRef(
            int(args.scalar(11)).to_bytes(8, "little"),
            int(args.scalar(12)),
            int(args.scalar(13)),
        )
    )
    if phase == 0:
        _reset(orch, domain, node, nodes)
        return
    if phase == 2:
        _event_expected(actor, target, write, serial, nodes)
        _verify(orch, domain, node, nodes)
        return
    local = domain.host(node)
    if write:
        orch.copy_to_host_domain(domain, _pattern(100 + serial, SPAN), offset=SOURCE, node=node)
    if actor < nodes and target == node:
        data = orch.copy_from_host_domain(domain, SPAN, offset=SOURCE if write else TARGET, node=node)
        orch.copy_to_host_domain(domain, data, offset=TARGET if write else RESULT, node=node)
        return
    rank = 2 * node if actor < nodes else actor - nodes
    chip = domain.member(rank).local_worker_id
    operand = _pools(domain)[target]
    source = local if write else operand
    destination = operand if write else local
    chip_args = TaskArgs()
    for buffer, offset, tag in (
        (source, SOURCE if write else TARGET, TensorArgType.INPUT),
        (destination, TARGET if write else RESULT, TensorArgType.OUTPUT_EXISTING),
    ):
        tensor = domain.map_view(
            buffer.tensor([COUNT], DataType.FLOAT32, byte_offset=offset),
            domain_rank=rank,
            access=AccessMode.READ if tag == TensorArgType.INPUT else AccessMode.WRITE,
        )
        chip_args.add_tensor(tensor, tag, transfer=TensorTransfer.NONE)
    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little") for i in range(7, 11))
    _KEEPALIVE[:] = [chip_args]
    orch.submit_next_level(chip_handle or get_inner_handle(digest.hex()), chip_args, cfg, worker=chip)


def _build_kernel(platform="a2a3"):
    compiler = KernelCompiler(platform=platform)
    kernels = Path(__file__).resolve().parent / "kernels"
    includes = list(compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer"))
    includes.append(str(compiler.project_root / "src" / "common"))
    binary = compiler.compile_incore(
        source_path=str(kernels / "aiv" / "copy.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=includes,
    )
    orchestration = compiler.compile_orchestration(
        runtime_name="tensormap_and_ringbuffer",
        source_path=str(kernels / "orchestration" / "copy.cpp"),
    )
    signature = [ArgDirection.IN, ArgDirection.OUT]
    return ChipCallable.build(
        signature=signature,
        func_name="host_domain_copy",
        config_name="host_domain_copy_config",
        binary=orchestration,
        children=[
            (
                0,
                CoreCallable.build(
                    signature=signature,
                    binary=binary if platform.endswith("sim") else extract_text_section(binary),
                ),
            )
        ],
    )


def controller(context):
    executors = context.topology.executors
    nodes = len(executors)
    if nodes not in (1, 2) or any(len(spec.device_ids) != 2 for spec in executors):
        raise ValueError("requires one or two executor nodes, each with two NPU chips")
    if tuple(spec.worker_id for spec in executors) != tuple(range(nodes)):
        raise ValueError("executor worker ids must be consecutive from zero")
    worker = context.create_worker(num_sub_workers=0)
    kernel = worker.register(_build_kernel())
    route = worker.register(RemoteCallable(MODULE + ":node_orch"), workers=list(range(nodes)))
    worker.init()
    run_matrix(worker, nodes, kernel, route)


def run_matrix(worker, nodes, kernel, route):
    """Run the same acceptance flow on an initialized L4 and registered L3 route."""
    saved = []
    keepalive = []
    cfg = CallConfig()
    cfg.aicpu_thread_num = 2
    try:

        def allocate(orch, _args, _cfg):
            domain = orch.get_or_allocate_domain(
                name="host-domain-matrix",
                members=[(node, chip) for node in range(nodes) for chip in (0, 1)],
                window_size=WINDOW,
                buffers=[CommBufferSpec("payload", "float32", WINDOW // 4, WINDOW)],
                host_windows={node: HostWindowSpec(nbytes=HOST_WINDOW, numa_id=0) for node in range(nodes)},
                retain_after_run=True,
            )
            if len({pool.identity for pool in _pools(domain)}) != 3 * nodes:
                raise AssertionError("domain pool identities are not distinct")
            if any(domain.host(node).address_space != AddressSpace.HOST for node in range(nodes)):
                raise AssertionError("Host backing must retain its physical HOST location")
            for node in range(nodes):
                marker = _pattern(900 + node, SPAN)
                orch.copy_to_host_domain(domain, marker, node=node, offset=TARGET)
                if orch.copy_from_host_domain(domain, SPAN, node=node, offset=TARGET) != marker:
                    raise AssertionError(f"public Host staging mismatch node={node}")
            saved.append(domain)

        worker.run(allocate, config=cfg)
        domain = saved[0]

        def step(phase, actor, target, write, serial):
            actor_node = actor if actor < nodes else (actor - nodes) // 2

            def submit(orch, _args, config):
                current = orch.get_domain(domain.ref)
                keepalive.clear()
                for node in range(nodes):
                    if phase == 1 and node != actor_node:
                        continue
                    args = TaskArgs()
                    for value in (phase, node, nodes, actor, target, write, serial):
                        args.add_scalar(value)
                    for offset in range(0, 32, 8):
                        args.add_scalar(int.from_bytes(kernel.digest[offset : offset + 8], "little"))
                    for value in (
                        int.from_bytes(current.ref.owner_instance_id, "little"),
                        current.ref.domain_id,
                        current.ref.generation,
                    ):
                        args.add_scalar(value)
                    keepalive.append(args)
                    orch.submit_next_level(route, args, config, worker=node)

            worker.run(submit, config=cfg)

        checks = 0
        for actor in range(3 * nodes):
            for target in range(3 * nodes):
                for write in (0, 1):
                    checks += 1
                    step(0, actor, target, write, checks)
                    step(1, actor, target, write, checks)
                    step(2, actor, target, write, checks)
                    method = (
                        "CPU_LOCAL"
                        if actor < nodes and actor == target
                        else "DEVICE_ASSISTED"
                        if actor < nodes
                        else "KERNEL"
                    )
                    print(
                        f"CASE_PASS actor={actor} pool={target} "
                        f"operation={'WRITE' if write else 'READ'} method={method}",
                        flush=True,
                    )
        handoffs = 0
        for node in range(nodes):
            step(0, node, node, 1, 1000)
            actors = [node, nodes + 2 * node, nodes + 2 * node + 1]
            if nodes == 2:
                actors.append(nodes + 2 * (node ^ 1))
            for actor in actors:
                for write in (0, 1):
                    handoffs += 1
                    step(1, actor, node, write, 1000 + handoffs)
                    step(2, actor, node, write, 1000 + handoffs)
            # The owning CPU consumes the last NPU's bytes without resetting the allocation.
            step(1, node, node, 0, 2000)
            step(2, node, node, 0, 2000)
        domain.release()
        if not domain.freed or worker.live_domains:
            raise AssertionError("domain release did not complete")
        print(
            json.dumps(
                {
                    "result": "MATRIX_PASS",
                    "nodes": nodes,
                    "domain_pools": 3 * nodes,
                    "checks": checks,
                    "cpu_local": 2 * nodes,
                    "cpu_device_assisted": 2 * nodes * (3 * nodes - 1),
                    "kernel": 12 * nodes * nodes,
                    "handoff_steps": handoffs + nodes,
                    "released": True,
                }
            ),
            flush=True,
        )
    finally:
        keepalive.clear()
        if saved and not saved[0].released:
            saved[0].release()


_SPARSE = {}


def sparse_node_orch(orch, args, cfg):
    phase, node, nodes, row, chip = (int(args.scalar(i)) for i in range(5))
    ref = CommDomainRef(int(args.scalar(5)).to_bytes(8, "little"), int(args.scalar(6)), int(args.scalar(7)))
    domain = orch.get_domain(ref)
    worker = orch._worker
    if phase == 0:
        for index in range(128):
            orch.copy_to_host_domain(
                domain, struct.pack(f"<{N}f", *([float(node * 1000 + index)] * N)), offset=index * SIZE, node=node
            )
        _SPARSE[node] = [worker.create_buffer(size) for size in (4, SIZE, SIZE, SIZE)]
        return
    ids, independent, consumed, independent_result = _SPARSE[node]
    peer = (node + 1) % nodes
    if phase == 2:
        assert struct.unpack_from(f"<{N}f", consumed.shm.buf) == (float(peer * 1000 + row + 1),) * N
        assert struct.unpack_from(f"<{N}f", independent_result.shm.buf) == (12.0,) * N
        print(f"SPARSE_PASS node={node} chip={chip} source_node={peer} row={row}", flush=True)
        return
    if phase == 3:
        for buffer in _SPARSE.pop(node):
            worker.release_buffer(buffer)
        return
    ids.shm.buf[:4] = struct.pack("<I", row)
    independent.shm.buf[:SIZE] = struct.pack(f"<{N}f", *([11.0] * N))
    rank = next(m.domain_rank for m in domain.members if m.node_worker_id == node and m.local_worker_id == chip)
    call = TaskArgs()
    call.add_tensor(
        domain.map_view(
            domain.host(peer).tensor((128 * N,), DataType.FLOAT32), domain_rank=rank, access=AccessMode.READ
        ),
        TensorArgType.INPUT,
        transfer=TensorTransfer.NONE,
    )
    call.add_tensor(ids.tensor((1,), DataType.UINT32), TensorArgType.INPUT)
    call.add_tensor(independent.tensor((N,), DataType.FLOAT32), TensorArgType.INPUT)
    call.add_tensor(
        domain.map_view(
            domain.member(rank).buffers["payload"].tensor((N,), DataType.FLOAT32),
            domain_rank=rank,
            access=AccessMode.WRITE,
        ),
        TensorArgType.OUTPUT_EXISTING,
        transfer=TensorTransfer.NONE,
    )
    call.add_tensor(consumed.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
    call.add_tensor(independent_result.tensor((N,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little") for i in range(8, 12))
    _KEEPALIVE[:] = [call]
    orch.submit_next_level(get_inner_handle(digest.hex()), call, cfg, worker=chip)


def sparse_controller(context):
    nodes = len(context.topology.executors)
    worker = context.create_worker(num_sub_workers=0, enable_sdma=True)
    kernel = worker.register(build_sparse_sdma("a2a3"))
    route = worker.register(RemoteCallable(MODULE + ":sparse_node_orch"), workers=list(range(nodes)))
    saved, keepalive = [], []
    try:
        worker.init()

        def allocate(orch, _args, _cfg):
            saved.append(
                orch.allocate_domain(
                    name="host-fabric-sparse",
                    members=[(n, c) for n in range(nodes) for c in (0, 1)],
                    window_size=SIZE,
                    buffers=[CommBufferSpec("payload", "float32", N, SIZE)],
                    host_windows={n: HostWindowSpec(128 * SIZE) for n in range(nodes)},
                    retain_after_run=True,
                )
            )

        worker.run(allocate)
        domain = saved[0]

        def step(phase, row=0, chip=0):
            def dispatch(orch, _args, cfg):
                current = orch.get_domain(domain.ref)
                keepalive.clear()
                for node in range(nodes):
                    args = TaskArgs()
                    for value in (
                        phase,
                        node,
                        nodes,
                        row,
                        chip,
                        int.from_bytes(current.ref.owner_instance_id, "little"),
                        current.ref.domain_id,
                        current.ref.generation,
                    ):
                        args.add_scalar(value)
                    for offset in range(0, 32, 8):
                        args.add_scalar(int.from_bytes(kernel.digest[offset : offset + 8], "little"))
                    keepalive.append(args)
                    orch.submit_next_level(route, args, cfg, worker=node)

            worker.run(dispatch)

        step(0)
        for chip in (0, 1):
            for row in (0, 127, 7, 7):
                step(1, row, chip)
                step(2, row, chip)
        step(3)
        domain.release()
        assert domain.freed
        print("SPARSE_MATRIX_PASS: real consumers and independent compute; overlap timing not measured", flush=True)
    finally:
        worker.close()


def run(
    *,
    local_host,
    remote_host,
    python_executable,
    local_devices,
    remote_devices,
    mpirun_path,
    launcher_family,
    sparse=False,
):
    pairs = [(local_host, local_devices)]
    if remote_host is not None:
        pairs.append((remote_host, remote_devices))
    executors = []
    for index, (host, devices) in enumerate(pairs):
        ids = [int(value) for value in devices.split(",")]
        if len(ids) != 2 or min(ids) < 0 or ids[0] == ids[1]:
            raise ValueError("each executor requires two distinct non-negative device ids")
        executors.append(
            {
                "rank": index + 1,
                "worker_id": index,
                "host": host,
                "platform": "a2a3",
                "runtime": "tensormap_and_ringbuffer",
                "device_ids": ids,
                "comm_profile": "a3-fabric-v1",
            }
        )
    topology = {
        "controller_rank": 0,
        "controller_host": local_host,
        "startup_timeout_s": 240.0,
        "session_timeout_s": 180.0,
        "heartbeat_interval_s": 1.0,
        "max_pending_frame_bytes": 64 * 1024 * 1024,
        "launcher_args": ["-wdir", "/tmp"],
        "executor_ranks": executors,
    }
    with tempfile.TemporaryDirectory(prefix="simpler-host-domain-") as directory:
        path = Path(directory) / "topology.json"
        path.write_text(json.dumps(topology), encoding="utf-8")
        return run_supervisor(
            str(path),
            MODULE + (":sparse_controller" if sparse else ":controller"),
            python_executable=python_executable,
            mpirun_path=mpirun_path,
            launcher_family=launcher_family,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", required=True, help="numeric IP of controller and first executor")
    parser.add_argument("--remote-host", help="numeric IP of second executor; omit for one-node control")
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--local-devices", default="0,1")
    parser.add_argument("--remote-devices", default="0,1")
    parser.add_argument("--mpirun-path", default="mpirun")
    parser.add_argument("--launcher-family", choices=("auto", "openmpi", "mpich"), default="auto")
    parser.add_argument("--sparse", action="store_true", help="PTO SDMA sparse fetch with real consumers")
    return run(**vars(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
