#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Extra Host process reads and writes remote HBM during an L3 control callback."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import struct
import tempfile
from pathlib import Path

from simpler.comm_domain import CommDomainRef
from simpler.mpi_direct_supervisor import run_supervisor
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, HostWindowSpec, TaskArgs, TensorArgType
from simpler.worker import RemoteCallable

MODULE = "examples.workers.l4.host_consumer_mpi_direct.main"
WINDOW = 4096
HOST_WINDOW = 8192
STAGING = 128
CAPACITY = 64
TARGET = 128
PATCH = TARGET + 16
TARGET_RANK = 2
SNAPSHOT = 4096
_STALE = []


def _pattern(rank):
    return bytes((rank * 17 + offset) % 251 for offset in range(WINDOW))


def _host_pattern(node):
    return bytes([0xA0 + node]) * HOST_WINDOW


def _consume(payload, owner_pid):
    child_pid = os.getpid()
    if child_pid == owner_pid:
        raise AssertionError("Host consumer must run in a distinct process")
    before = bytearray(CAPACITY)
    payload.read(TARGET, before)
    expected = bytearray(_pattern(TARGET_RANK)[TARGET : TARGET + CAPACITY])
    if before != expected:
        raise AssertionError("SUB did not read the remote HBM initialization")
    proof = struct.pack("<QQ", child_pid, owner_pid)
    payload.write(PATCH, proof)
    expected[16:32] = proof
    after = bytearray(CAPACITY)
    payload.read(TARGET, after)
    if after != expected:
        raise AssertionError("SUB read-after-write or guard mismatch")
    _STALE[:] = [payload]
    print(f"HOST_CONSUMER_IO child_pid={child_pid} owner_pid={owner_pid} target_rank={TARGET_RANK}", flush=True)


def _check_revoked(_payload):
    if not _STALE:
        raise AssertionError("prior SUB attachment is missing")
    try:
        _STALE[0].read(0, bytearray(1))
    except RuntimeError:
        return
    raise AssertionError("prior SUB payload remains usable after call_host returned")


def node_orch(orch, args, _cfg):
    phase, node = int(args.scalar(0)), int(args.scalar(1))
    ref = CommDomainRef(int(args.scalar(2)).to_bytes(8, "little"), int(args.scalar(3)), int(args.scalar(4)))
    domain = orch.get_domain(ref)
    host, part = domain.host(node), domain.local_host()
    if phase == 0:
        for member in domain.members:
            if member.node_worker_id == node:
                part.write(SNAPSHOT, _pattern(member.domain_rank))
                orch.copy_domain(
                    member.buffers["payload"],
                    host,
                    worker=member.local_worker_id,
                    nbytes=WINDOW,
                    src_offset=SNAPSHOT,
                )
        part.write(0, _host_pattern(node))
        return
    if phase == 1:
        if node != 0:
            raise AssertionError("only node A serves the Host consumer")
        target = domain.member(TARGET_RANK).buffers["payload"]
        owner_pid = os.getpid()

        def consume(payload):
            _consume(payload, owner_pid)

        kwargs = dict(domain=ref, staging_offset=STAGING, staging_nbytes=CAPACITY, sub_worker=0, worker=0)
        orch.call_host(consume, target, request_limit=3, **kwargs)
        orch.call_host(_check_revoked, target, request_limit=1, **kwargs)
        return
    observed_host = bytearray(HOST_WINDOW)
    part.read(0, observed_host)
    expected_host = _host_pattern(node)
    if node == 0:
        if (
            observed_host[:STAGING] != expected_host[:STAGING]
            or observed_host[STAGING + CAPACITY :] != expected_host[STAGING + CAPACITY :]
        ):
            raise AssertionError("consumer staging modified Host pool guards")
    elif observed_host != expected_host:
        raise AssertionError("remote provider Host pool was changed by the consumer")
    for member in domain.members:
        if member.node_worker_id != node:
            continue
        orch.copy_domain(
            host, member.buffers["payload"], worker=member.local_worker_id, nbytes=WINDOW, dst_offset=SNAPSHOT
        )
        actual = bytearray(WINDOW)
        part.read(SNAPSHOT, actual)
        expected = bytearray(_pattern(member.domain_rank))
        if member.domain_rank == TARGET_RANK:
            child_pid, owner_pid = struct.unpack_from("<QQ", actual, PATCH)
            if not child_pid or not owner_pid or child_pid == owner_pid:
                raise AssertionError("provider did not observe a distinct SUB process proof")
            expected[PATCH : PATCH + 16] = struct.pack("<QQ", child_pid, owner_pid)
            print(f"PROVIDER_VERIFY child_pid={child_pid} owner_pid={owner_pid}", flush=True)
        if actual != expected:
            raise AssertionError(f"provider HBM data or guard mismatch rank={member.domain_rank}")


def run_case(worker, route):
    """Exercise an initialized two-node L4 using its registered L3 callback."""
    saved, keepalive = [], []
    cfg = CallConfig()

    def submit(orch, domain, phase, nodes, config):
        keepalive.clear()
        for node in nodes:
            args = TaskArgs()
            args.add_tensor(domain.host(node).tensor([HOST_WINDOW], DataType.UINT8), TensorArgType.INPUT)
            for value in (
                phase,
                node,
                int.from_bytes(domain.ref.owner_instance_id, "little"),
                domain.domain_id,
                domain.generation,
            ):
                args.add_scalar(value)
            keepalive.append(args)
            orch.submit_next_level(route, args, config, worker=node)

    def initialize(orch, _args, config):
        domain = orch.get_or_allocate_domain(
            name="host-consumer",
            members=[(node, chip) for node in (0, 1) for chip in (0, 1)],
            window_size=WINDOW,
            buffers=[CommBufferSpec("payload", "uint8", WINDOW, WINDOW)],
            host_windows={node: HostWindowSpec(0, HOST_WINDOW) for node in (0, 1)},
            retain_after_run=True,
        )
        saved.append(domain)
        submit(orch, domain, 0, (0, 1), config)

    worker.run(initialize, config=cfg)
    domain = saved[0]
    try:
        for phase, nodes in ((1, (0,)), (2, (0, 1))):

            def step(orch, _args, config):
                current = orch.get_domain(domain.ref)
                submit(orch, current, phase, nodes, config)

            worker.run(step, config=cfg)
        domain.release()
        if not domain.freed or worker.live_domains:
            raise AssertionError("retained domain release did not complete")
        print(
            json.dumps(
                {
                    "result": "HOST_CONSUMER_PASS",
                    "nodes": 2,
                    "chips": 4,
                    "sub_processes_per_node": 1,
                    "io_requests": 3,
                    "revoked": True,
                    "released": True,
                }
            ),
            flush=True,
        )
    finally:
        if not domain.freed:
            domain.release()


def controller(context):
    executors = context.topology.executors
    if len(executors) != 2 or any(len(spec.device_ids) != 2 or spec.num_sub_workers != 1 for spec in executors):
        raise ValueError("requires two executors, each with two chips and one SUB Host process")
    if tuple(spec.worker_id for spec in executors) != (0, 1):
        raise ValueError("executor worker ids must be 0 and 1")
    worker = context.create_worker(num_sub_workers=0)
    route = worker.register(RemoteCallable(MODULE + ":node_orch"), workers=[0, 1])
    worker.init()
    run_case(worker, route)


def run(
    local_host,
    remote_host,
    python_executable,
    local_devices="0,1",
    remote_devices="0,1",
    mpirun_path="mpirun",
    launcher_family="auto",
):
    if ipaddress.ip_address(local_host) == ipaddress.ip_address(remote_host):
        raise ValueError("remote Host consumer validation requires two distinct executor IPs")
    executors = []
    for node, (host, devices) in enumerate(((local_host, local_devices), (remote_host, remote_devices))):
        ids = [int(value) for value in devices.split(",")]
        if len(ids) != 2 or min(ids) < 0 or ids[0] == ids[1]:
            raise ValueError("each executor requires two distinct non-negative device ids")
        executors.append(
            dict(
                rank=node + 1,
                worker_id=node,
                host=host,
                platform="a2a3",
                runtime="tensormap_and_ringbuffer",
                device_ids=ids,
                comm_profile="a3-fabric-v1",
                num_sub_workers=1,
            )
        )
    topology = dict(
        controller_rank=0,
        controller_host=local_host,
        startup_timeout_s=240.0,
        session_timeout_s=180.0,
        heartbeat_interval_s=1.0,
        max_pending_frame_bytes=64 * 1024 * 1024,
        launcher_args=["-wdir", "/tmp"],
        executor_ranks=executors,
    )
    with tempfile.TemporaryDirectory(prefix="simpler-host-consumer-") as directory:
        path = Path(directory) / "topology.json"
        path.write_text(json.dumps(topology), encoding="utf-8")
        return run_supervisor(
            str(path),
            MODULE + ":controller",
            python_executable=python_executable,
            mpirun_path=mpirun_path,
            launcher_family=launcher_family,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", required=True, help="numeric IP of controller and executor A")
    parser.add_argument("--remote-host", required=True, help="numeric IP of executor B")
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--local-devices", default="0,1")
    parser.add_argument("--remote-devices", default="0,1")
    parser.add_argument("--mpirun-path", default="mpirun")
    parser.add_argument("--launcher-family", choices=("auto", "openmpi", "mpich"), default="auto")
    return run(**vars(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
