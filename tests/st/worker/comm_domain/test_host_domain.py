# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host DRAM and HBM access use the same matrix as the MPI hardware runner."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from simpler.comm_domain import CommDomainRef
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from examples.workers.l4.host_domain_mpi_direct.main import _build_kernel, node_orch, run_matrix
from tests.st.worker.comm_domain import test_domain_tensor_remote as remote_domain

_RUNTIME = "tensormap_and_ringbuffer"
tcp_daemon = remote_domain.tcp_daemon


def _node_orch_with_attachment_probe(orch, args, cfg, chip_handle=None):
    node_orch(orch, args, cfg, chip_handle=chip_handle)
    if int(args.scalar(0)) != 0:
        return
    node = int(args.scalar(1))
    domain = orch.get_domain(
        CommDomainRef(int(args.scalar(11)).to_bytes(8, "little"), int(args.scalar(12)), int(args.scalar(13)))
    )
    view = orch.get_global_domain(domain.domain_id)
    row = view.attachments
    target = next((m for m in domain.members if m.node_worker_id != node), domain.members[0])
    try:
        view.attachments = ()
        with pytest.raises(RuntimeError, match="materialized domain attachment"):
            orch.copy_domain(target.buffers["payload"], domain.host(node), worker=0, nbytes=16)
    finally:
        view.attachments = row


def _simulate_distinct_nodes(monkeypatch, worker):
    snapshot = worker._endpoint_topology_snapshot

    def topology():
        value = snapshot()
        return replace(
            value,
            entries=tuple(
                replace(entry, node_identity=entry.path.split("/")[1]) if "/" in entry.path else entry
                for entry in value.entries
            ),
        )

    monkeypatch.setattr(worker, "_endpoint_topology_snapshot", topology)


def _assert_matrix(output, nodes, checks, handoffs):
    summary = json.loads(output.splitlines()[-1])
    assert output.count("CASE_PASS ") == checks
    assert summary["result"] == "MATRIX_PASS" and summary["checks"] == checks
    assert summary["nodes"] == nodes and summary["domain_pools"] == 3 * nodes
    assert summary["handoff_steps"] == handoffs and summary["released"]
    for node in range(nodes):
        assert f"ATTACHMENT_PLAN node={node} remote={2 * (nodes - 1)}" in output


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.parametrize(
    "nodes,checks,handoffs",
    [
        pytest.param(1, 18, 7, marks=pytest.mark.device_count(2), id="1"),
        pytest.param(2, 72, 18, marks=pytest.mark.device_count(4), id="2"),
    ],
)
def test_host_domain_full_access_matrix(st_platform, st_device_ids, capsys, monkeypatch, nodes, checks, handoffs):
    worker = Worker(level=4, num_sub_workers=0)
    _simulate_distinct_nodes(monkeypatch, worker)
    chips = []
    try:
        callable_obj = _build_kernel(st_platform)
        for node in range(nodes):
            child = Worker(
                level=3,
                platform=st_platform,
                runtime=_RUNTIME,
                device_ids=[int(device) for device in st_device_ids[2 * node : 2 * node + 2]],
                num_sub_workers=0,
                comm_profile="sim",
                global_device_ranks=[2 * node, 2 * node + 1],
            )
            chips.append(child.register(callable_obj))
            assert worker.add_worker(child) == node
        kernel = worker.register(callable_obj)

        def forward(orch, args, cfg):
            _node_orch_with_attachment_probe(orch, args, cfg, chip_handle=chips[int(args.scalar(1))])

        route = worker.register(forward)
        worker.init()
        run_matrix(worker, nodes, kernel, route)
        _assert_matrix(capsys.readouterr().out, nodes, checks, handoffs)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(4)
def test_host_domain_full_access_matrix_through_tcp(tcp_daemon, st_platform, st_device_ids, capsys, monkeypatch):
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=30)
    _simulate_distinct_nodes(monkeypatch, worker)
    try:
        for node in range(2):
            assert (
                worker.add_remote_worker(
                    RemoteWorkerSpec(
                        endpoint=tcp_daemon,
                        platform=st_platform,
                        runtime=_RUNTIME,
                        device_ids=tuple(int(device) for device in st_device_ids[2 * node : 2 * node + 2]),
                        comm_profile="sim",
                        global_device_ranks=(2 * node, 2 * node + 1),
                    )
                )
                == node
            )
        kernel = worker.register(_build_kernel(st_platform))
        route = worker.register(RemoteCallable(__name__ + ":_node_orch_with_attachment_probe"), workers=[0, 1])
        worker.init()
        run_matrix(worker, 2, kernel, route)
        _assert_matrix(capsys.readouterr().out, 2, 72, 18)
    finally:
        worker.close()
