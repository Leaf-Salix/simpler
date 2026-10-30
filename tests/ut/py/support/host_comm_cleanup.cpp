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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using aclError = int;
using aclrtDrvMemHandle = void *;
constexpr int ACL_SUCCESS = 0, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC = 2;
constexpr int ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT = 0;
constexpr int COMM_GLOBAL_DOMAIN_VERSION = 1, COMM_GLOBAL_DOMAIN_PROFILE_A3_FABRIC = 2;
struct CommContext {};
struct VmmWindow {
    bool live = true;
};
struct aclrtMemFabricHandle {
    char bytes[128];
};
struct CommGlobalDomainDescriptor {
    uint32_t version = 1, profile = 2, reserved = 0;
    uint32_t handle_size = 128;
    uint64_t mapping_size = 4096;
    char handle[128]{};
};
static std::vector<std::string> calls, logs;
static std::string fail_at;
static bool persistent = false, throw_import = false, hbm_failure = false;
static int step(const char *name) {
    calls.emplace_back(name);
    if (fail_at == name) {
        if (!persistent) fail_at.clear();
        return 701;
    }
    return 0;
}
template <typename... Args>
void log_message(const char *format, Args... args) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf), format, args...);
    logs.emplace_back(buf);
}
#define LOG_ERROR(...) log_message(__VA_ARGS__)
#define LOG_WARN(...) log_message(__VA_ARGS__)
static void record_first_error(int status, int *error) {
    if (!*error) *error = status;
}
static int aclrtUnmapMem(void *) { return step("unmap"); }
static int aclrtReleaseMemAddress(void *) { return step("release_va"); }
static int aclrtFreePhysical(void *) { return step("free_handle"); }
static int aclrtFree(void *) { return step("free_ctx"); }
static int release_vmm_window(VmmWindow *window) {
    if (!window->live) return 0;
    window->live = false;
    int status = step("hbm");
    return hbm_failure ? 702 : status;
}
static int release_vmm_windows(std::vector<VmmWindow> *windows) {
    int status = 0;
    for (auto &window : *windows)
        record_first_error(release_vmm_window(&window), &status);
    return status;
}
static int aclrtReserveMemAddressNoUCMemory(void **out, size_t, size_t, void *address, uint64_t) {
    int status = step("reserve");
    if (!status) *out = address;
    return status;
}
constexpr void *RTLD_DEFAULT = nullptr;
static void *dlsym(void *, const char *) {
    return fail_at == "symbol" ? nullptr : reinterpret_cast<void *>(&aclrtReserveMemAddressNoUCMemory);
}
static int aclrtMemImportFromShareableHandleV2(aclrtMemFabricHandle *, int, int, void **out) {
    if (throw_import) throw std::runtime_error("injected import exception");
    int status = step("import");
    if (!status) *out = reinterpret_cast<void *>(2);
    return status;
}
static int aclrtMapMem(void *, size_t, size_t, void *, int) { return step("map"); }
// PRODUCTION_BODIES
struct Handle {
    int rank = 0;
    std::string rootinfo_path, run_token;
    std::unordered_map<uint64_t, std::unique_ptr<DomainAllocation>> domain_allocations;
};
using CommHandle = Handle *;
static std::string domain_barrier_tag(uint64_t, const char *) { return "release"; }
static bool file_barrier(const std::string &, int, int, const std::string &, const std::string &) {
    return step("barrier") == 0;
}
// TEST_CASES
static std::unique_ptr<DomainAllocation> allocation() {
    auto owner = std::make_unique<DomainAllocation>();
    owner->nranks = 1;
    owner->host_base = reinterpret_cast<void *>(1);
    owner->host_handle = reinterpret_cast<void *>(2);
    owner->host_mapped = true;
    owner->device_ctx = reinterpret_cast<CommContext *>(3);
    owner->peer_windows.emplace_back();
    return owner;
}
static bool called(const char *name) { return std::find(calls.begin(), calls.end(), name) != calls.end(); }
int main() {
    for (const char *failure : {"unmap", "release_va", "free_handle"}) {
        calls.clear();
        fail_at = failure;
        persistent = true;
        auto owner = allocation();
        assert(release_domain_windows(owner.get()) == 701);
        assert(called("hbm"));
        if (fail_at == "unmap") {
            assert(!called("release_va") && !called("free_handle"));
            assert(owner->host_mapped);
        }
        if (fail_at == "release_va") assert(!called("free_handle"));
        fail_at.clear();
        persistent = false;
        assert(release_domain_host(owner.get()) == 0);
        assert(!owner->host_base && !owner->host_handle && !owner->host_mapped);
    }
    for (const char *failure : {"unmap", "release_va", "free_handle"}) {
        Handle handle;
        handle.domain_allocations.emplace(7, allocation());
        calls.clear();
        fail_at = failure;
        hbm_failure = true;
        assert(comm_release_domain_windows(&handle, 7, 1, 0) == 701);
        assert(called("barrier") && called("hbm") && called("free_ctx"));
        assert(handle.domain_allocations.empty());
        assert(std::count(calls.begin(), calls.end(), "barrier") == 1);
        hbm_failure = false;
        assert(std::count(calls.begin(), calls.end(), failure) == 2);
    }
    {
        Handle handle;
        handle.domain_allocations.emplace(7, allocation());
        calls.clear();
        fail_at = "unmap";
        persistent = true;
        assert(comm_release_domain_windows(&handle, 7, 1, 0) == 701);
        assert(comm_release_domain_windows(&handle, 7, 1, 0) == -1);
        assert(std::count(calls.begin(), calls.end(), "barrier") == 1);
        assert(!called("release_va") && !called("free_handle"));
        persistent = false;
        fail_at.clear();
    }
    calls.clear();
    { auto owner = allocation(); }
    assert(called("unmap") && called("release_va") && called("free_handle"));
    calls.clear();
    fail_at = "unmap";
    persistent = true;
    { auto owner = allocation(); }
    assert(!called("release_va") && !called("free_handle"));
    persistent = false;
    fail_at.clear();
    for (const char *failure : {"import", "reserve", "map", "symbol", "exception"}) {
        Handle handle;
        handle.domain_allocations.emplace(7, std::make_unique<DomainAllocation>());
        CommGlobalDomainDescriptor descriptor;
        uint64_t result = 123;
        calls.clear();
        logs.clear();
        fail_at = failure;
        persistent = true;
        throw_import = fail_at == "exception";
        assert(comm_attach_domain_host(&handle, 7, &descriptor, &result) != 0);
        assert(result == 0 && !logs.empty());
        assert(std::any_of(logs.begin(), logs.end(), [](const auto &entry) {
            return entry.find("allocation_id=7 stage=") != std::string::npos;
        }));
        fail_at.clear();
        persistent = false;
        throw_import = false;
        handle.domain_allocations.clear();
        if (std::string(failure) == "reserve" || std::string(failure) == "map") assert(called("free_handle"));
        if (std::string(failure) == "map") assert(called("release_va") && !called("unmap"));
    }
    Handle handle;
    handle.domain_allocations.emplace(7, std::make_unique<DomainAllocation>());
    CommGlobalDomainDescriptor descriptor;
    uint64_t result = 123;
    descriptor.version = 99;
    logs.clear();
    assert(comm_attach_domain_host(&handle, 7, &descriptor, &result) == -1);
    assert(logs.back().find("version=99") != std::string::npos);
    descriptor.version = 1;
    assert(comm_attach_domain_host(&handle, 7, &descriptor, &result) == 0);
    assert(result != 0);
}
