# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Platform-specific Host backing; domain identity and lifetime live in host_domain."""

from _task_interface import _HostDomainMemory

from .buffer import create_host_shared_buffer
from .comm_endpoints import AdapterProfile
from .global_comm_domain import (
    GLOBAL_DOMAIN_PROFILE_A3_FABRIC,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_PROFILE_SIM,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainDescriptor,
)


class SimHostBacking:
    profile = AdapterProfile.HOST_SHM_MAP

    def __init__(self):
        self.shared = None

    def allocate(self, spec, identity, device, rank_count):
        self.shared = create_host_shared_buffer(
            spec.nbytes,
            bytes(identity.owner_instance_id),
            identity.buffer_id,
            generation=identity.generation,
        )
        return self.shared.base, GlobalDomainDescriptor(
            GLOBAL_DOMAIN_VERSION,
            GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_SIM],
            0,
            rank_count,
            spec.nbytes,
            self.shared.body,
        )

    def release(self):
        if self.shared is not None:
            self.shared.close()


class A3HostBacking:
    profile = AdapterProfile.HOST_VMM_MAP

    def __init__(self):
        self.native = _HostDomainMemory()

    def allocate(self, spec, identity, device, rank_count):
        size, token = self.native.allocate(device, spec.numa_id, spec.nbytes)
        return self.native.base, GlobalDomainDescriptor(
            GLOBAL_DOMAIN_VERSION,
            GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC],
            0,
            rank_count,
            size,
            token,
        )

    def release(self):
        self.native.release()


def host_backing_type(platform):
    """Select before any allocation. Unsupported platforms fail without side effects."""
    backends = {"a2a3sim": SimHostBacking, "a2a3": A3HostBacking}
    try:
        return backends[platform]
    except KeyError:
        raise ValueError(f"Host domain backing is not implemented for {platform!r}") from None


def host_attachment_profile(descriptor: GlobalDomainDescriptor) -> AdapterProfile:
    profiles = {
        GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_SIM]: AdapterProfile.HOST_SHM_MAP,
        GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC]: AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT,
    }
    try:
        return profiles[descriptor.profile_id]
    except KeyError:
        raise ValueError("unsupported Host attachment profile") from None
