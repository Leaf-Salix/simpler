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
// Test-only ACL surface for exercising the real nanobind HostDomainMemory owner.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool live_physical = false, live_va = false, mapped = false, exported = false;
static int step(const char *name) {
    FILE *log = std::fopen(std::getenv("FABRIC_TEST_LOG"), "a");
    std::fprintf(log, "%s\n", name);
    std::fclose(log);
    const char *fail = std::getenv("FABRIC_TEST_FAIL");
    return fail && std::strcmp(fail, name) == 0 ? 701 : 0;
}
extern "C" {
int aclInit(const char *) {
    if (int rc = step("init")) return rc;
    const char *result = std::getenv("FABRIC_TEST_INIT_RESULT");
    return result ? std::atoi(result) : 0;
}
int aclrtSetDevice(int) { return step("device"); }
int aclrtGetDevice(int *device) {
    *device = 0;
    return 0;
}
const char *aclrtGetSocName() { return "Ascend910_9391"; }
int aclrtDeviceGetBareTgid(int32_t *pid) {
    *pid = 12345;
    return 0;
}
int aclrtMemcpy(void *, size_t, const void *, size_t, int) { return 0; }
int aclrtMemGetAllocationGranularity(void *, int, size_t *size) {
    if (int rc = step("granularity")) return rc;
    *size = 4096;
    return 0;
}
int aclrtMallocPhysical(void **handle, size_t, const void *, uint64_t) {
    if (int rc = step("physical")) return rc;
    *handle = std::malloc(1);
    live_physical = true;
    return 0;
}
int aclrtReserveMemAddress(void **base, size_t size, size_t, void *, uint64_t) {
    if (int rc = step("reserve")) return rc;
    *base = std::malloc(size);
    live_va = true;
    return 0;
}
int aclrtMapMem(void *, size_t, size_t, void *, uint64_t) {
    if (int rc = step("map")) return rc;
    mapped = true;
    return 0;
}
int aclrtMemExportToShareableHandleV2(void *, uint64_t flags, int type, void *token) {
    if (flags != 0 || type != 2 || !mapped || !live_physical) return 702;
    if (int rc = step("export")) return rc;
    std::memset(token, 0x5a, 128);
    exported = true;
    return 0;
}
int aclrtMemSetPidToShareableHandleV2(void *token, int type, int32_t *pids, size_t count) {
    if (!exported || !mapped || type != 2 || count != 2 || pids[0] != 12345 || pids[1] != 23456 ||
        static_cast<unsigned char *>(token)[0] != 0x5a)
        return 703;
    return step("authorize");
}
int aclrtUnmapMem(void *) {
    if (int rc = step("unmap")) return rc;
    if (!mapped) return 704;
    mapped = false;
    return 0;
}
int aclrtReleaseMemAddress(void *base) {
    if (int rc = step("release_va")) return rc;
    if (mapped || !live_va) return 705;
    std::free(base);
    live_va = false;
    return 0;
}
int aclrtFreePhysical(void *handle) {
    if (int rc = step("free")) return rc;
    if (live_va || mapped || !live_physical) return 706;
    std::free(handle);
    live_physical = false;
    return 0;
}
int fabric_test_live() { return int(live_physical) + int(live_va) + int(mapped); }
int aclrtMemSetAccess(void *, size_t, void *, size_t) { return 0; }
int aclrtMemExportToShareableHandle(void *, int, uint64_t, uint64_t *) { return 0; }
int aclrtMemImportFromShareableHandle(uint64_t, int32_t, void **) { return 0; }
int aclrtReserveMemAddressNoUCMemory(void **, size_t, size_t, void *, uint64_t) { return 0; }
}
