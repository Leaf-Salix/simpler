# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    mint_owner_instance_id,
)
from simpler.comm_domain import domain_mapping_ref, domain_mapping_target
from simpler.comm_endpoints import DEVICE_AICPU, AdapterProfile
from simpler.task_interface import (
    ArgDirection,
    DataType,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
    _materialize_chip_task_args,
)


@pytest.fixture
def mapping():
    identity = CanonicalIdentity(mint_owner_instance_id(), 1, 1)
    buffer = Buffer(
        identity=identity,
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=64,
        base=4096,
    )
    registry = ImportRegistry(ImportContext(DEVICE_AICPU))
    for access in (AccessMode.READWRITE, AccessMode.READ, AccessMode.WRITE):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=domain_mapping_ref(0, access),
            access=access,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    return buffer, registry


@pytest.mark.parametrize(
    "access,tag,direction,allowed",
    [
        (AccessMode.READ, TensorArgType.INPUT, ArgDirection.IN, True),
        (AccessMode.READ, TensorArgType.OUTPUT, ArgDirection.OUT, False),
        (AccessMode.READ, TensorArgType.INOUT, ArgDirection.INOUT, False),
        (AccessMode.WRITE, TensorArgType.OUTPUT, ArgDirection.OUT, True),
        (AccessMode.WRITE, TensorArgType.INPUT, ArgDirection.OUT, True),
        (AccessMode.WRITE, TensorArgType.INPUT, ArgDirection.IN, False),
        (AccessMode.READWRITE, TensorArgType.INOUT, ArgDirection.INOUT, True),
    ],
)
def test_selected_mapping_grant(mapping, access, tag, direction, allowed):
    buffer, registry = mapping
    native = buffer.tensor((16,), DataType.FLOAT32)
    view = native._with_mapping_ref(domain_mapping_ref(0, access))
    args = TaskArgs()
    args.add_tensor(view, tag, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda index: direction)
    if allowed:
        assert _materialize_chip_task_args(args, registry, target).tensor(0).data == 8192
    else:
        with pytest.raises(ValueError, match="mapping grant"):
            _materialize_chip_task_args(args, registry, target)
    assert native.mapping_ref == 0 and view.buffer == native.buffer


def test_callable_direction_cannot_be_weakened_by_input_tag(mapping):
    buffer, registry = mapping
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(domain_mapping_ref(0, AccessMode.READ))
    args = TaskArgs()
    args.add_tensor(view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
    target = SimpleNamespace(sig_count=1, sig=lambda index: ArgDirection.OUT)
    with pytest.raises(ValueError, match="mapping grant"):
        _materialize_chip_task_args(args, registry, target)


def test_mapping_rejects_grant_change_and_unregistered_reference(mapping):
    buffer, registry = mapping
    with pytest.raises(ValueError, match="conflicts"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=domain_mapping_ref(0, AccessMode.READ),
            access=AccessMode.READWRITE,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(domain_mapping_ref(1, AccessMode.READ))
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(view)
    registry.unregister(buffer.identity)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(view._with_mapping_ref(domain_mapping_ref(0, AccessMode.READ)))


@pytest.mark.parametrize("rank", [-1, 64])
def test_mapping_rank_limits(rank):
    with pytest.raises(ValueError, match="mapping rank"):
        domain_mapping_ref(rank, AccessMode.READ)


def test_mapping_reference_profiles():
    for rank in (0, 63):
        for access in (AccessMode.READ, AccessMode.WRITE, AccessMode.READWRITE):
            assert domain_mapping_target(domain_mapping_ref(rank, access)) == (rank, access)


def test_read_mapping_preflight_precedes_other_argument_materialization(mapping, monkeypatch):
    buffer, registry = mapping
    args = TaskArgs()
    args.add_tensor(buffer.tensor((16,), DataType.FLOAT32), transfer=TensorTransfer.H2D)
    view = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(domain_mapping_ref(0, AccessMode.READ))
    args.add_tensor(view, TensorArgType.OUTPUT, transfer=TensorTransfer.NONE)
    monkeypatch.setattr(registry, "materialize", lambda _: pytest.fail("must reject before ordinary import"))
    with pytest.raises(ValueError, match="mapping grant"):
        registry.materialize_args(args)


def test_attachment_cannot_widen_source_grant(mapping):
    buffer, registry = mapping
    buffer.access = AccessMode.READ
    registry = ImportRegistry(ImportContext(DEVICE_AICPU))
    with pytest.raises(ValueError, match="exceeds source grant"):
        registry.register_attachment(
            buffer.to_descriptor(),
            8192,
            mapping_ref=domain_mapping_ref(2, AccessMode.READWRITE),
            access=AccessMode.READWRITE,
            profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
            live=lambda: True,
        )
