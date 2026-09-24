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

#include <cstdint>

#include "host_build_graph/runtime_types.h"
#include "task_args.h"

namespace hbg {

// HBG kernel mode borrows caller-owned Device tensors. Host orchestration may
// inspect their descriptors, but it cannot read or write their storage.
enum class KernelExternalTensorStatus : uint8_t {
    Ok = 0,
    InvalidCounts,
    InvalidTensor,
    NonDeviceTensor,
    UnsupportedStrideFamily,
};

inline bool valid_external_tensor_span(const ChipTensor &tensor, uint64_t *extent_out = nullptr) noexcept {
    if (tensor.buffer.addr == 0 || tensor.buffer.addr >= HEAP_VIRTUAL_BASE || tensor.buffer.size == 0 ||
        tensor.buffer.size > UINT64_MAX - tensor.buffer.addr || tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS ||
        static_cast<uint8_t>(tensor.dtype) >= static_cast<uint8_t>(DataType::DATA_TYPE_NUM))
        return false;
    const uint64_t element_bytes = get_element_size(tensor.dtype);
    if (element_bytes == 0) return false;
    uint64_t extent = 1;
    for (uint32_t i = 0; i < tensor.ndims; ++i) {
        const uint64_t shape = tensor.shapes[i];
        const uint64_t stride = tensor.strides[i];
        if (shape == 0 || stride == 0 || (shape - 1) > (UINT64_MAX - extent) / stride) return false;
        extent += (shape - 1) * stride;
    }
    if (tensor.start_offset > tensor.buffer.size / element_bytes ||
        extent > tensor.buffer.size / element_bytes - tensor.start_offset)
        return false;
    if (extent_out != nullptr) *extent_out = extent;
    return true;
}

// V1 accepts dense row-major tensors and row-major views with padding between
// outer rows. The innermost dimension is unit-stride; every outer stride must
// cover the full next dimension. Transposes, broadcasts and stepped innermost
// slices need a separate specialization and are rejected before Host build.
inline bool supported_external_stride_family(const ChipTensor &tensor) noexcept {
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS || tensor.strides[tensor.ndims - 1] != 1) return false;
    for (uint32_t i = tensor.ndims - 1; i > 0; --i) {
        const uint64_t inner = tensor.strides[i];
        const uint64_t shape = tensor.shapes[i];
        if (shape != 0 && inner > UINT64_MAX / shape) return false;
        if (tensor.strides[i - 1] < inner * shape) return false;
    }
    return true;
}

inline KernelExternalTensorStatus validate_kernel_external_tensors(const ChipStorageTaskArgs &args) noexcept {
    const int32_t tensor_count = args.tensor_count();
    const int32_t scalar_count = args.scalar_count();
    if (tensor_count < 0 || tensor_count > CHIP_MAX_TENSOR_ARGS || scalar_count < 0 ||
        scalar_count > CHIP_MAX_SCALAR_ARGS || tensor_count + scalar_count > CHIP_MAX_TENSOR_ARGS)
        return KernelExternalTensorStatus::InvalidCounts;
    for (int32_t i = 0; i < tensor_count; ++i) {
        const ChipTensor &tensor = args.tensor(i);
        if (tensor.address_space != AddressSpace::DEVICE) return KernelExternalTensorStatus::NonDeviceTensor;
        if (!valid_external_tensor_span(tensor)) return KernelExternalTensorStatus::InvalidTensor;
        if (!supported_external_stride_family(tensor)) return KernelExternalTensorStatus::UnsupportedStrideFamily;
    }
    return KernelExternalTensorStatus::Ok;
}

}  // namespace hbg
