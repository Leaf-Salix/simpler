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

#include <acl/acl.h>
#include <atomic>
#include "runtime_c_api.h"
#include "task_args.h"
#include "common/host_log_state.h"

namespace {
std::atomic<int> launches{0};
}
extern "C" {
int simpler_host_log_bind_state(SimplerHostLogState *) { return 0; }
DeviceContextHandle create_device_context() { return new int(-1); }
void destroy_device_context(DeviceContextHandle ctx) { delete static_cast<int *>(ctx); }
int finalize_device(DeviceContextHandle) { return 0; }
int simpler_kernel_mode_supported(DeviceContextHandle) { return 1; }
size_t committed_device_memory_ctx(DeviceContextHandle) { return 0; }
int simpler_kernel_mode_init(
    DeviceContextHandle ctx, int device, const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *, size_t,
    const CallConfig *, uint64_t
) {
    int actual = -1;
    const int rc = aclrtGetDevice(&actual);
    if (rc || actual != device) return PTO_RUNTIME_ERR_INVALID_STATE;
    *static_cast<int *>(ctx) = device;
    return 0;
}
int simpler_kernel_mode_prepare_callable(DeviceContextHandle, int32_t, const void *, size_t) { return 0; }
int test_launch_count() { return launches.load(); }
int simpler_kernel_mode_launch(DeviceContextHandle, int32_t, const void *raw, void *stream) {
    const auto &args = *static_cast<const ChipStorageTaskArgs *>(raw);
    if (args.tensor_count() != 2 || args.scalar_count() != 1) return PTO_RUNTIME_ERR_INTERNAL;
    const auto &src = args.tensor(0);
    const auto &dst = args.tensor(1);
    const size_t bytes = args.scalar(0);
    const uint64_t src_offset = src.start_offset * get_element_size(src.dtype);
    const uint64_t dst_offset = dst.start_offset * get_element_size(dst.dtype);
    if (src_offset > src.buffer.size || dst_offset > dst.buffer.size || bytes > src.buffer.size - src_offset ||
        bytes > dst.buffer.size - dst_offset)
        return PTO_RUNTIME_ERR_INTERNAL;
    ++launches;
    if (bytes == 0) return 0;
    return aclrtMemcpyAsync(
        reinterpret_cast<void *>(dst.buffer.addr + dst_offset), dst.buffer.size - dst_offset,
        reinterpret_cast<const void *>(src.buffer.addr + src_offset), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
        static_cast<aclrtStream>(stream)
    );
}
}
