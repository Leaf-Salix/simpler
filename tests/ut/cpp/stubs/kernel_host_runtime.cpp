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

#include "runtime_c_api.h"
#include "common/host_log_state.h"
#include "task_args.h"
#include <atomic>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<int> launches{0}, destroys{0}, finalizes{0};
std::atomic<int> init_result{0}, close_result{0}, launch_result{0};
std::atomic<int> creates{0}, prepares{0}, prepare_result{0};
std::atomic<uint64_t> scalar_value{0};
std::atomic<uint64_t> generation_value{0};
std::atomic<uintptr_t> init_context{0}, prepare_context{0}, launch_context{0};
std::atomic<int32_t> prepare_cid{-1}, launch_cid{-1};
}  // namespace
extern "C" {
int simpler_host_log_bind_state(SimplerHostLogState *) { return 0; }
void kernel_test_reset() {
    launches = 0;
    destroys = 0;
    finalizes = 0;
    init_result = 0;
    close_result = 0;
    launch_result = 0;
    creates = 0;
    prepares = 0;
    prepare_result = 0;
    scalar_value = 0;
    generation_value = 0;
    init_context = 0;
    prepare_context = 0;
    launch_context = 0;
    prepare_cid = -1;
    launch_cid = -1;
}
void kernel_test_errors(int init, int close, int launch) {
    init_result = init;
    close_result = close;
    launch_result = launch;
}
int kernel_test_launches() { return launches; }
int kernel_test_destroys() { return destroys; }
int kernel_test_finalizes() { return finalizes; }
int kernel_test_creates() { return creates; }
int kernel_test_prepares() { return prepares; }
void kernel_test_prepare_error(int error) { prepare_result = error; }
uint64_t kernel_test_generation() { return generation_value; }
uintptr_t kernel_test_init_context() { return init_context; }
uintptr_t kernel_test_prepare_context() { return prepare_context; }
uintptr_t kernel_test_launch_context() { return launch_context; }
int32_t kernel_test_prepare_cid() { return prepare_cid; }
int32_t kernel_test_launch_cid() { return launch_cid; }
void kernel_test_release_on_thread(void *data, void (*release)(void *) noexcept) {
    std::thread worker([=] {
        release(data);
    });
    worker.join();
}
uint64_t kernel_test_scalar() { return scalar_value; }
DeviceContextHandle create_device_context() {
    ++creates;
    return new int(0);
}
void destroy_device_context(DeviceContextHandle ctx) {
    ++destroys;
    delete static_cast<int *>(ctx);
}
int finalize_device(DeviceContextHandle) {
    ++finalizes;
    return close_result;
}
int simpler_kernel_mode_supported(DeviceContextHandle) { return 1; }
size_t committed_device_memory_ctx(DeviceContextHandle) { return 4096; }
int simpler_kernel_mode_init(
    DeviceContextHandle ctx, int, const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *, size_t,
    const CallConfig *, uint64_t generation
) {
    init_context = reinterpret_cast<uintptr_t>(ctx);
    generation_value = generation;
    return init_result;
}
int simpler_kernel_mode_prepare_callable(DeviceContextHandle ctx, int32_t cid, const void *, size_t) {
    ++prepares;
    prepare_context = reinterpret_cast<uintptr_t>(ctx);
    prepare_cid = cid;
    return prepare_result;
}
int simpler_kernel_mode_launch(DeviceContextHandle ctx, int32_t cid, const void *raw, void *) {
    ++launches;
    launch_context = reinterpret_cast<uintptr_t>(ctx);
    launch_cid = cid;
    if (launch_result == -12345) throw std::runtime_error("injected native exception");
    auto &args = *static_cast<const ChipStorageTaskArgs *>(raw);
    if (args.scalar_count()) scalar_value = args.scalar(0);
    return launch_result;
}
}
