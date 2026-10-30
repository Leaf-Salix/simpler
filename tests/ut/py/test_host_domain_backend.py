# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host registration belongs to the selected platform backend."""

from unittest.mock import MagicMock

import _task_interface
import pytest
from simpler import host_domain_backend as backend


def test_a3_registration_uses_runtime_identity_and_checks_authorization(monkeypatch):
    native = MagicMock()
    monkeypatch.setattr(backend, "_HostDomainMemory", lambda: native)
    monkeypatch.setattr(_task_interface, "_fabric_process_id", lambda: 12345)
    owner = backend.host_backing_type("a2a3")()
    owner.authorize([backend.host_process_identity("a2a3")])
    native.authorize.assert_called_once_with([12345])


def test_optional_host_identity_does_not_enable_unsupported_backing(monkeypatch):
    def unexpected_fabric_call():
        raise AssertionError("non-Fabric communicator requested a Fabric identity")

    monkeypatch.setattr(_task_interface, "_fabric_process_id", unexpected_fabric_call)
    assert backend.host_process_identity("a2a3sim") == 0
    backend.host_backing_type("a2a3sim")().authorize([0, 0])
    assert backend.host_process_identity("a5") == 0
    with pytest.raises(ValueError, match="not implemented"):
        backend.host_backing_type("a5")


def test_authorization_targets_preserve_real_identities_and_reject_invalid_profiles():
    assert backend.host_authorization_targets("a3-fabric-v1", (101, 202)) == (101, 202)
    assert backend.host_authorization_targets("sim", (0, 0)) == (1, 1)
    for identities in ((), (0,), (101, -1)):
        with pytest.raises(ValueError, match="invalid Fabric process IDs"):
            backend.host_authorization_targets("a3-fabric-v1", identities)
    with pytest.raises(ValueError, match="not implemented"):
        backend.host_authorization_targets("unknown", (101,))
