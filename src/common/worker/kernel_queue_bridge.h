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

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "tensor.h"

// Internal same-build bridge between two Python extensions. No STL object or
// allocation crosses this boundary; objects are destroyed by their producer.
namespace simpler::kernel {
inline constexpr const char *KERNEL_QUEUE_CAPSULE = "simpler.kernel_queue.v1";
struct KernelQueueTicket {
    void *opaque;
    void (*release)(void *) noexcept;
    int (*invoke)(void *, void *, int (*)(void *) noexcept, void *) noexcept;
};
struct KernelQueueBridge {
    uint32_t version;
    uint32_t struct_size;
    int32_t device;
    void *opaque;
    int (*accept)(void *, int32_t, const ChipTensor *, size_t, const uint64_t *, size_t, KernelQueueTicket *) noexcept;
};
static_assert(std::is_trivially_copyable_v<KernelQueueBridge>);
static_assert(std::is_trivially_copyable_v<KernelQueueTicket>);
}  // namespace simpler::kernel
