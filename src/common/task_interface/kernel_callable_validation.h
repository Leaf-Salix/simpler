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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "callable.h"
#include "callable_protocol.h"

namespace simpler::kernel {

inline bool valid_kernel_callable_span(const void *image, size_t bytes) noexcept {
    const auto address = reinterpret_cast<uintptr_t>(image);
    return address != 0 && address % alignof(ChipCallable) == 0 && bytes >= sizeof(ChipCallable) &&
           bytes <= UINTPTR_MAX - address;
}

// The readable image uses canonical contiguous child offsets; no child header
// is read until its complete span lies inside the supplied image.
inline bool valid_kernel_callable_image(const void *callable_image, size_t callable_size) {
    if (!valid_kernel_callable_span(callable_image, callable_size)) {
        return false;
    }
    const auto *callable = static_cast<const ChipCallable *>(callable_image);
    if (callable->sig_count_ < 0 || callable->sig_count_ > CHIP_MAX_TENSOR_ARGS || callable->child_count_ < 0 ||
        callable->child_count_ > 1024 || callable->func_name_len_ >= CALLABLE_FUNC_NAME_MAX ||
        callable->config_name_len_ >= CALLABLE_FUNC_NAME_MAX ||
        callable->func_name_[callable->func_name_len_] != '\0' ||
        callable->config_name_[callable->config_name_len_] != '\0') {
        return false;
    }

    const size_t header_size = offsetof(ChipCallable, storage_);
    const size_t storage_size = callable_size - header_size;
    size_t used = callable->binary_size_;
    if (used > storage_size) return false;

    for (int32_t i = 0; i < callable->child_count_; ++i) {
        /* A func_id outside the device function table, or one repeated within
           this image, would have the device consumer overwrite a mapping it
           built earlier in the same invocation. */
        const int32_t func_id = callable->child_func_ids_[i];
        const auto *seen_end = callable->child_func_ids_ + i;
        if (func_id < 0 || func_id >= KERNEL_MAX_FUNC_ID ||
            std::find(callable->child_func_ids_, seen_end, func_id) != seen_end) {
            return false;
        }
        if (used > SIZE_MAX - (CALLABLE_ALIGN - 1)) return false;
        const size_t aligned = (used + CALLABLE_ALIGN - 1) & ~(static_cast<size_t>(CALLABLE_ALIGN) - 1);
        const size_t offset = callable->child_offsets_[i];
        if (aligned < used || offset != aligned || offset > storage_size ||
            CoreCallable::binary_data_offset() > storage_size - offset) {
            return false;
        }
        const auto *child = reinterpret_cast<const CoreCallable *>(callable->storage_ + offset);
        if (child->sig_count_ < 0 || child->sig_count_ > CORE_MAX_TENSOR_ARGS) {
            return false;
        }
        const size_t child_header = CoreCallable::binary_data_offset();
        const size_t child_binary = child->binary_size_;
        if (child_binary > storage_size - offset - child_header) return false;
        used = offset + child_header + child_binary;
    }
    return used == storage_size;
}

}  // namespace simpler::kernel
