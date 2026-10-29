# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Platform-specific Host backing; domain identity and lifetime live in host_domain."""

import os

from .buffer import create_host_shared_buffer
from .comm_endpoints import AdapterProfile
from .global_comm_domain import (
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_PROFILE_SIM,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainDescriptor,
)


class SharedHostBacking:
    profile = AdapterProfile.HOST_SHM_MAP
    attachment_profile = AdapterProfile.HOST_SHM_MAP

    def __init__(self):
        self.shared = None

    def allocate(self, spec, identity, device, rank_count):
        page_size = os.sysconf("SC_PAGE_SIZE")
        mapping_size = (spec.nbytes + page_size - 1) // page_size * page_size
        self.shared = create_host_shared_buffer(
            mapping_size,
            bytes(identity.owner_instance_id),
            identity.buffer_id,
            generation=identity.generation,
        )
        return self.shared.base, GlobalDomainDescriptor(
            GLOBAL_DOMAIN_VERSION,
            GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_SIM],
            0,
            rank_count,
            mapping_size,
            self.shared.body,
        )

    def release(self):
        if self.shared is not None:
            self.shared.close()


class A3HostBacking(SharedHostBacking):
    attachment_profile = AdapterProfile.HOST_SVM_MAP

    def allocate(self, spec, identity, device, rank_count):
        if spec.numa_id != 0:
            raise ValueError("HostRegister backing uses OS placement; explicit NUMA placement is unsupported")
        return super().allocate(spec, identity, device, rank_count)


def host_backing_type(platform):
    """Select before any allocation. Unsupported platforms fail without side effects."""
    backends = {"a2a3sim": SharedHostBacking, "a2a3": A3HostBacking}
    try:
        return backends[platform]
    except KeyError:
        raise ValueError(f"Host domain backing is not implemented for {platform!r}") from None


def host_attachment_profile(descriptor: GlobalDomainDescriptor, platform: str) -> AdapterProfile:
    if descriptor.profile_id != GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_SIM]:
        raise ValueError("HostRegister attachment requires a local shared-memory descriptor")
    return host_backing_type(platform).attachment_profile
