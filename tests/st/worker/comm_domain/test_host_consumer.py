# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A distinct SUB consumer actively accesses another node's HBM."""

import json

import pytest
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from examples.workers.l4.host_consumer_mpi_direct.main import MODULE, node_orch, run_case
from tests.st.worker.comm_domain import test_domain_tensor_remote as remote_domain
from tests.st.worker.comm_domain.test_host_domain import _simulate_distinct_nodes

_RUNTIME = "tensormap_and_ringbuffer"
tcp_daemon = remote_domain.tcp_daemon


def _assert_result(output):
    summary = json.loads(output.splitlines()[-1])
    assert summary == {
        "result": "HOST_CONSUMER_PASS",
        "nodes": 2,
        "chips": 4,
        "sub_processes_per_node": 1,
        "io_requests": 3,
        "revoked": True,
        "released": True,
    }


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(4)
def test_sub_consumer_accesses_remote_hbm(st_platform, st_device_ids, capsys, monkeypatch):
    worker = Worker(level=4, num_sub_workers=0)
    _simulate_distinct_nodes(monkeypatch, worker)
    try:
        for node in range(2):
            child = Worker(
                level=3,
                platform=st_platform,
                runtime=_RUNTIME,
                device_ids=[int(device) for device in st_device_ids[2 * node : 2 * node + 2]],
                num_sub_workers=1,
                comm_profile="sim",
                global_device_ranks=[2 * node, 2 * node + 1],
            )
            assert worker.add_worker(child) == node
        route = worker.register(node_orch)
        worker.init()
        run_case(worker, route)
        _assert_result(capsys.readouterr().out)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(4)
def test_sub_consumer_accesses_remote_hbm_through_tcp(tcp_daemon, st_platform, st_device_ids, capsys, monkeypatch):
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
                        num_sub_workers=1,
                        comm_profile="sim",
                        global_device_ranks=(2 * node, 2 * node + 1),
                    )
                )
                == node
            )
        route = worker.register(RemoteCallable(MODULE + ":node_orch"), workers=[0, 1])
        worker.init()
        run_case(worker, route)
        _assert_result(capsys.readouterr().out)
    finally:
        worker.close()
