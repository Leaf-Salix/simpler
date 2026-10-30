# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile shared production VMM/staged-release bodies against faulting ACL calls."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_host_global_cleanup_faults(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    source = (root / "src/a2a3/platform/onboard/host/comm_hccl.cpp").read_text()

    def section(start, end):
        return source[source.index(start) : source.index(end, source.index(start))]

    fixture = (Path(__file__).parent / "support/host_comm_cleanup.cpp").read_text()
    declarations, cases = fixture.split("// PRODUCTION_BODIES")
    bodies = [
        section("static aclError release_vmm_window(", "// The driver owns reservations"),
        section('extern "C" int comm_global_domain_release(', 'extern "C" int comm_destroy('),
    ]
    unit = tmp_path / "host_comm_cleanup.cpp"
    unit.write_text(declarations + "\n".join(bodies) + cases)
    executable = tmp_path / "host_comm_cleanup"
    subprocess.run([compiler, "-std=c++17", str(unit), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
