# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simulation coverage for the two-host MPI coexistence runner."""

import pytest
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from examples.workers.l4.host_domain_mpi_direct.coexistence import MODULE, run_coexistence
from examples.workers.l4.host_domain_mpi_direct.main import _build_kernel
from tests.st.worker.comm_domain import test_domain_tensor_remote as remote_domain
from tests.st.worker.comm_domain.test_host_domain import _simulate_distinct_nodes

_RUNTIME = "tensormap_and_ringbuffer"
tcp_daemon = remote_domain.tcp_daemon


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(4)
@pytest.mark.parametrize("release_order", [("host", "ep", "tp"), ("ep", "tp", "host")])
def test_host_and_npu_domains_coexist(tcp_daemon, st_platform, st_device_ids, monkeypatch, release_order):
    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=30)
    _simulate_distinct_nodes(monkeypatch, worker)
    try:
        for node in range(2):
            worker.add_remote_worker(
                RemoteWorkerSpec(
                    endpoint=tcp_daemon,
                    platform=st_platform,
                    runtime=_RUNTIME,
                    device_ids=tuple(int(d) for d in st_device_ids[2 * node : 2 * node + 2]),
                    comm_profile="sim",
                    global_device_ranks=(2 * node, 2 * node + 1),
                )
            )
        kernel = worker.register(_build_kernel(st_platform))
        route = worker.register(RemoteCallable(MODULE + ":_copy_peer"), workers=[0, 1])
        worker.init()

        run_coexistence(worker, kernel, route, release_order)
    finally:
        worker.close()
