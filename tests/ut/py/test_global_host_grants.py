# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Global Host views select endpoint grants while sharing one physical attachment."""

import ctypes
from types import SimpleNamespace

import pytest
from _task_interface import read_args_from_blob
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    Tensor,
    mint_owner_instance_id,
)
from simpler.comm_domain import CommDomainRef, domain_mapping_ref, domain_mapping_target
from simpler.comm_endpoints import DEVICE_AICORE, HOST_CPU, AdapterProfile
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_PROFILE_SIM,
    GlobalDomainCommand,
    GlobalDomainMember,
    GlobalDomainPhase,
)
from simpler.task_interface import (
    ArgDirection,
    ChipDomainContext,
    CommDomainHandle,
    DataType,
    TensorArgType,
    _materialize_chip_task_args,
)
from simpler.worker import Worker, _GlobalNodeDomainState

from tests.ut.py.test_worker._wire_blob import encode_blob


def source_buffer():
    return Buffer(
        identity=CanonicalIdentity(mint_owner_instance_id(), 1, 1),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=64,
    )


def global_handle(buffer):
    members = (GlobalDomainMember(0, 3, 0, 0), GlobalDomainMember(0, 7, 1, 1))
    contexts = {worker: ChipDomainContext("test", rank, 2, 100, 200, 64, {}) for rank, worker in enumerate((3, 7))}
    return CommDomainHandle(
        name="test", workers=(3, 7), contexts=contexts, _release_fn=None, members=members, host_buffers={0: buffer}
    ), members


def test_global_view_grants_share_one_mapping_and_preserve_rw_selector():
    buffer = source_buffer()
    domain, _ = global_handle(buffer)
    tensor = buffer.tensor((16,), DataType.FLOAT32)
    registry = ImportRegistry(ImportContext(DEVICE_AICORE))
    for access in (AccessMode.READWRITE, AccessMode.READ, AccessMode.WRITE):
        view = domain.map_view(tensor, domain_rank=1, access=access)
        assert domain_mapping_target(view.mapping_ref) == (1, access)
        assert view.buffer == tensor.buffer
        registry.register_borrowed(
            buffer.to_descriptor(), 4096, mapping=view.mapping_ref, access=access, live=lambda: True
        )
        assert registry.resolve_tensor(view).base == 4096
    assert domain.map_view(tensor, domain_rank=1).mapping_ref == 2
    with pytest.raises(ValueError, match="grant"):
        registry.require_mapping_access(
            domain.map_view(tensor, domain_rank=1, access=AccessMode.READ), AccessMode.WRITE
        )


def test_global_dispatch_checks_target_and_selected_grant():
    buffer = source_buffer()
    domain, members = global_handle(buffer)
    registry = ImportRegistry(ImportContext(HOST_CPU))
    registry.register_borrowed(buffer.to_descriptor(), 0, live=lambda: True)
    worker = Worker(level=3, num_sub_workers=0)
    worker._global_node_domains[1] = _GlobalNodeDomainState(
        command=GlobalDomainCommand(GlobalDomainPhase.COMMIT, 1, 1, "test", GLOBAL_DOMAIN_PROFILE_SIM, 64, members, ()),
        imports=registry,
        node_worker_id=0,
    )
    source = buffer.tensor((16,), DataType.FLOAT32)
    read = domain.map_view(source, domain_rank=1, access=AccessMode.READ)
    worker._domain_mapping_buffer(read, 7, 1)
    with pytest.raises(ValueError, match="grant"):
        worker._domain_mapping_buffer(read, 7, 2)
    with pytest.raises(ValueError, match="another node or chip"):
        worker._domain_mapping_buffer(read, 3, 1)
    with pytest.raises(ValueError, match="invalid domain mapping"):
        worker._domain_mapping_buffer(source._with_mapping_ref(193), 7, 1)


def test_consumer_signature_not_transport_tag_controls_global_grant():
    buffer = source_buffer()
    registry = ImportRegistry(ImportContext(DEVICE_AICORE))
    source = buffer.tensor((16,), DataType.FLOAT32)
    target = SimpleNamespace(sig_count=1, sig=lambda _: ArgDirection.OUT)
    for access, allowed in ((AccessMode.READ, False), (AccessMode.WRITE, True)):
        selector = domain_mapping_ref(1, access)
        registry.register_borrowed(buffer.to_descriptor(), 4096, mapping=selector, access=access, live=lambda: True)
        blob = ctypes.create_string_buffer(encode_blob([source._with_mapping_ref(selector)]))
        args = read_args_from_blob(ctypes.addressof(blob), len(blob))
        # Transport reconstruction defaults tags to INPUT; the actual consumer writes.
        assert args.tag(0) == TensorArgType.INPUT
        if allowed:
            _materialize_chip_task_args(args, registry, target)
        else:
            with pytest.raises(ValueError, match="grant"):
                _materialize_chip_task_args(args, registry, target)


@pytest.mark.parametrize("writing", [False, True])
def test_borrowed_copy_checks_owner_before_native_routing(monkeypatch, writing):
    worker = Worker(level=3, num_sub_workers=0)
    owner = mint_owner_instance_id()
    state = SimpleNamespace(
        phase=GlobalDomainPhase.COMMIT,
        view=object(),
        command=SimpleNamespace(owner_instance_id=owner, generation=1),
    )
    monkeypatch.setattr(worker, "_global_node_domains", {1: state})
    calls = []

    def copy(command, *, copy_to_device):
        calls.append((command, copy_to_device))
        return b"data"

    monkeypatch.setattr(worker, "_copy_global_domain_node", copy)

    def invoke(identity):
        handle = CommDomainHandle(
            name="test",
            _release_fn=None,
            domain_id=1,
            generation=1,
            members=(GlobalDomainMember(0, 0, 0, 0),),
            ref=CommDomainRef(identity, 1, 1),
        )
        if writing:
            worker._copy_to_global_domain(handle, 0, b"data", 0)
        else:
            worker._copy_from_global_domain(handle, 0, 4, 0)

    with pytest.raises(KeyError, match="no live grant"):
        invoke(mint_owner_instance_id())
    assert not calls
    invoke(owner)
    assert len(calls) == 1 and calls[0][1] is writing


def test_local_domain_slice_uses_the_canonical_device_descriptor():
    local = Buffer(
        identity=CanonicalIdentity(mint_owner_instance_id(), 2, 1),
        address_space=AddressSpace.DEVICE,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=64,
        base=4096,
        owner_worker_id=1,
    )
    descriptor = local.to_descriptor()
    registry = ImportRegistry(ImportContext(DEVICE_AICORE))
    registry.register_attachment(
        descriptor, 4096, mapping_ref=0, profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, live=lambda: True
    )
    for tensor in (local.tensor((16,), DataType.FLOAT32), Tensor(local, shapes=(16,), dtype=DataType.FLOAT32)):
        assert tensor.buffer == descriptor
        assert tensor.mapping_ref == 0
        assert registry.resolve_tensor(tensor).base == 4096
    with pytest.raises(ValueError, match="device endpoint"):
        ImportRegistry(ImportContext(HOST_CPU)).register_attachment(
            descriptor, 4096, mapping_ref=0, profile=AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, live=lambda: True
        )
    registry.unregister(descriptor.identity)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(local.tensor((16,), DataType.FLOAT32))

    host = source_buffer()
    members = (GlobalDomainMember(0, 1, 0, 0, {"slice": local}),)
    context = ChipDomainContext("local", 0, 1, 100, 4096, 64, {"slice": local})
    domain = CommDomainHandle(
        name="local",
        workers=(1,),
        contexts={1: context},
        members=members,
        host_buffers={0: host},
        _release_fn=lambda _: None,
    )
    view = domain.map_view(local.tensor((16,), DataType.FLOAT32), domain_rank=0)
    assert view.buffer == descriptor
    assert view.mapping_ref == domain_mapping_ref(0, AccessMode.READWRITE)
