# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the real HostDomainMemory state machine against faulting ACL calls."""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def fake_acl(tmp_path_factory):
    directory = tmp_path_factory.mktemp("host-fabric-acl")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = Path(__file__).parent / "support" / "host_fabric_acl.cpp"
    target = directory / "libascendcl.so"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(target)], check=True)
    return target


@pytest.mark.parametrize("operation", ["device", "granularity", "physical", "reserve", "map", "export", "authorize"])
def test_native_host_owner_partial_allocation_can_be_released(fake_acl, tmp_path, operation):
    _probe(fake_acl, tmp_path, operation, releasing=False)


@pytest.mark.parametrize("operation", ["device", "unmap", "release_va", "free"])
def test_native_host_owner_failed_release_retains_cleanup_for_retry(fake_acl, tmp_path, operation):
    _probe(fake_acl, tmp_path, operation, releasing=True)


def _probe(fake_acl, tmp_path, operation, *, releasing):
    log = tmp_path / "calls.log"
    env = dict(os.environ, FABRIC_TEST_LOG=str(log))
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[key] = str(fake_acl.parent) + os.pathsep + env.get(key, "")
    script = textwrap.dedent(
        """
        import ctypes
        import os
        import sys
        fake = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
        from _task_interface import _HostDomainMemory, _fabric_process_id
        assert _fabric_process_id() == 12345
        memory = _HostDomainMemory()
        operation, releasing = sys.argv[2], sys.argv[3] == '1'
        if releasing:
            size, token = memory.allocate(0, 0, 128)
            assert size == 4096 and token == bytes([0x5a]) * 128
            memory.authorize([12345, 23456])
        os.environ['FABRIC_TEST_FAIL'] = operation
        try:
            if releasing:
                memory.release()
            else:
                memory.allocate(0, 0, 128)
                memory.authorize([12345, 23456])
        except RuntimeError as error:
            assert '701' in str(error), error
        else:
            raise AssertionError('fault was not observed')
        del os.environ['FABRIC_TEST_FAIL']
        memory.release()
        assert fake.fabric_test_live() == 0
        memory.release()
        try:
            memory.authorize([12345, 23456])
        except ValueError:
            pass
        else:
            raise AssertionError('released owner authorized')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(fake_acl), operation, str(int(releasing))],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text().splitlines()
    # Successfully completed cleanup stages must not be replayed during retry.
    cleanup = [name for name in calls if name in ("unmap", "release_va", "free")]
    expected = {
        "device": [],
        "granularity": [],
        "physical": [],
        "reserve": ["free"],
        "map": ["release_va", "free"],
        "export": ["unmap", "release_va", "free"],
        "authorize": ["unmap", "release_va", "free"],
    }
    if releasing:
        expected_cleanup = ["unmap", "release_va", "free"]
        if operation in expected_cleanup:
            expected_cleanup.insert(expected_cleanup.index(operation), operation)
    else:
        expected_cleanup = expected[operation]
    assert cleanup == expected_cleanup


@pytest.mark.parametrize("init_result", [0, 100002, 701])
def test_native_host_owner_acl_initialization(fake_acl, tmp_path, init_result):
    log = tmp_path / "calls.log"
    env = dict(os.environ, FABRIC_TEST_LOG=str(log), FABRIC_TEST_INIT_RESULT=str(init_result))
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        env[key] = str(fake_acl.parent) + os.pathsep + env.get(key, "")
    script = textwrap.dedent(
        """
        import ctypes
        import os
        import sys
        fake = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
        from _task_interface import _HostDomainMemory
        memory = _HostDomainMemory()
        if int(os.environ['FABRIC_TEST_INIT_RESULT']) == 701:
            try:
                memory.allocate(0, 0, 128)
            except RuntimeError as error:
                assert 'aclInit failed with code 701' in str(error), error
            else:
                raise AssertionError('ACL initialization error was swallowed')
            assert fake.fabric_test_live() == 0
            os.environ['FABRIC_TEST_INIT_RESULT'] = '0'
        size, token = memory.allocate(0, 0, 128)
        assert size == 4096 and token == bytes([0x5a]) * 128
        for invalid in ([], [0], [12345, -1]):
            try:
                memory.authorize(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid authorization was accepted')
        memory.authorize([12345, 23456])
        memory.release()
        assert fake.fabric_test_live() == 0
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(fake_acl)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text().splitlines()
    assert calls.count("init") == (2 if init_result == 701 else 1)
    assert calls.count("physical") == 1
