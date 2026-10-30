/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#include <cassert>

#include "platform_comm/comm_sim.cpp"

int main() {
    constexpr auto profile = COMM_GLOBAL_DOMAIN_PROFILE_SIM_SHM;
    constexpr size_t bytes = 4096;
    CommGlobalDomainDescriptor descriptors[3]{};
    uint64_t local[2]{}, context = 0, base = 0;
    size_t size = 0;
    assert(COMM_GLOBAL_DOMAIN_VERSION == 2);
    for (uint32_t rank = 0; rank < 2; ++rank) {
        assert(comm_global_domain_prepare(7 + rank, rank, 2, bytes, profile, &descriptors[rank], &local[rank]) == 0);
    }
    descriptors[2] = descriptors[0];
    assert(comm_global_domain_import(7, descriptors, 1, &context) != 0);
    assert(comm_global_domain_import(7, descriptors, 3, &context) != 0);
    const auto name = "/host_domain_test_" + std::to_string(getpid());
    const auto *token = reinterpret_cast<const uint8_t *>(name.data());
    assert(comm_global_domain_import_host(7, profile, token, name.size(), bytes, &base) != 0);
    for (uint32_t rank = 0; rank < 2; ++rank) {
        assert(comm_global_domain_import(7 + rank, descriptors, 2, &context) == 0);
    }
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    assert(fd >= 0);
    assert(ftruncate(fd, bytes) == 0);
    // macOS rounds POSIX shared-memory backing sizes up to the system page size.
    struct stat backing{};
    assert(fstat(fd, &backing) == 0);
    assert(backing.st_size >= static_cast<off_t>(bytes));
    auto *owner = static_cast<uint8_t *>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    assert(owner != MAP_FAILED);
    owner[0] = 42;
    assert(comm_global_domain_import_host(7, 999, token, name.size(), bytes, &base) != 0);
    assert(comm_global_domain_import_host(7, profile, token, name.size() + 1, bytes, &base) != 0);
    assert(comm_global_domain_import_host(7, profile, token, name.size(), backing.st_size + 1, &base) != 0);
    for (uint32_t rank = 0; rank < 2; ++rank) {
        uint64_t host = 0;
        assert(comm_global_domain_import_host(7 + rank, profile, token, name.size(), bytes, &host) == 0);
        assert(*reinterpret_cast<uint8_t *>(host) == (rank == 0 ? 42 : 50));
        *reinterpret_cast<uint8_t *>(host) = static_cast<uint8_t>(50 + rank);
        assert(owner[0] == 50 + rank);
        assert(global_domain_allocations.at(7 + rank)->peer_mappings.size() == 2);
        assert(comm_global_domain_window(7 + rank, 2, &base, &size) != 0);
        for (uint32_t peer = 0; peer < 2; ++peer) {
            assert(comm_global_domain_window(7 + rank, peer, &base, &size) == 0);
            assert(size == bytes);
            *reinterpret_cast<uint8_t *>(base) = static_cast<uint8_t>(10 + peer);
            assert(*reinterpret_cast<uint8_t *>(local[peer]) == 10 + peer);
        }
    }
    for (uint32_t rank = 0; rank < 2; ++rank) {
        assert(comm_global_domain_release(7 + rank, 2) != 0);
        assert(comm_global_domain_release(7 + rank, 1) == 0);
        assert(global_domain_allocations.at(7 + rank)->peer_mappings.empty());
        assert(comm_global_domain_import_host(7 + rank, profile, token, name.size(), bytes, &base) != 0);
    }
    assert(owner[0] == 51);
    for (uint32_t rank = 0; rank < 2; ++rank)
        assert(comm_global_domain_release(7 + rank, 2) == 0);
    assert(global_domain_allocations.empty());
    assert(munmap(owner, bytes) == 0);
    assert(shm_unlink(name.c_str()) == 0);
}
