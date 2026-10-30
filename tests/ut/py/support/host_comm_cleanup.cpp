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
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
using aclError = int;
constexpr int ACL_SUCCESS = 0;
struct VmmWindow {
    void *base = nullptr, *handle = nullptr;
    uint64_t size = 4096, reservation_bytes = 4096;
    bool mapped = true;
};
struct DomainAllocation {
    bool host_enabled = true, views_released = false, release_failed = false;
    void *device_ctx = nullptr;
    VmmWindow local_window;
    std::vector<VmmWindow> peer_windows;
};
static std::unordered_map<uint64_t, std::unique_ptr<DomainAllocation>> global_domain_allocations;
static std::mutex global_domain_allocations_mutex;
static std::string fail_at;
static std::vector<std::string> calls;
static int step(const char *name) {
    calls.emplace_back(name);
    return fail_at == name ? 701 : 0;
}
static int aclrtFree(void *) { return step("context"); }
static int aclrtUnmapMem(void *) { return step("unmap"); }
static int aclrtReleaseMemAddress(void *) { return step("address"); }
static int aclrtFreePhysical(void *) { return step("handle"); }
// PRODUCTION_BODIES
static DomainAllocation &prepare(bool imported = true) {
    global_domain_allocations.clear();
    calls.clear();
    fail_at.clear();
    auto allocation = std::make_unique<DomainAllocation>();
    allocation->local_window.base = reinterpret_cast<void *>(1);
    allocation->local_window.handle = reinterpret_cast<void *>(2);
    if (imported) {
        allocation->device_ctx = reinterpret_cast<void *>(3);
        allocation->peer_windows.push_back({reinterpret_cast<void *>(4), reinterpret_cast<void *>(5)});
    }
    auto &result = *allocation;
    global_domain_allocations.emplace(7, std::move(allocation));
    return result;
}
int main() {
    for (bool imported : {false, true}) {
        auto &a = prepare(imported);
        assert(comm_global_domain_release(7, 0) != 0);
        assert(comm_global_domain_release(7, 2) != 0);
        assert(calls.empty());
        assert(comm_global_domain_release(7, 1) == 0);
        assert(a.views_released && a.peer_windows.empty());
        assert(a.local_window.base != nullptr);
        const auto completed = calls.size();
        assert(comm_global_domain_release(7, 1) == 0);
        assert(calls.size() == completed);
        assert(comm_global_domain_release(7, 2) == 0);
        assert(global_domain_allocations.empty());
        assert(comm_global_domain_release(7, 2) == 0);
    }
    for (const char *operation : {"context", "unmap", "address", "handle"}) {
        auto &a = prepare();
        fail_at = operation;
        assert(comm_global_domain_release(7, 1) != 0);
        assert(a.release_failed && !a.views_released && a.local_window.base != nullptr);
        const auto failed = calls.size();
        fail_at.clear();
        assert(comm_global_domain_release(7, 1) != 0);
        assert(comm_global_domain_release(7, 2) != 0);
        assert(calls.size() == failed);
    }
    for (const char *operation : {"unmap", "address", "handle"}) {
        auto &a = prepare();
        assert(comm_global_domain_release(7, 1) == 0);
        fail_at = operation;
        assert(comm_global_domain_release(7, 2) != 0);
        assert(a.release_failed && global_domain_allocations.count(7));
    }
}
