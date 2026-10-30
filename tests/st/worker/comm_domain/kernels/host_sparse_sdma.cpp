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

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "backend/sdma/sdma_completion_kernel.h"
#include "async_kernel_api.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *indices = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *ids = reinterpret_cast<__gm__ uint32_t *>(indices->buffer.addr) + indices->start_offset;
    const uint32_t row = ids[0];
    auto async_ctx = get_async_ctx(args);
    if (row >= 128) {
        pto2::detail::defer_error(async_ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    constexpr int elements = 128 * 128;
    using FlatShape = pto::Shape<1, 1, 1, 1, elements>;
    using FlatStride = pto::Stride<elements, elements, elements, elements, 1>;
    using Global = pto::GlobalTensor<float, FlatShape, FlatStride>;
    Global source(reinterpret_cast<__gm__ float *>(input->buffer.addr) + input->start_offset + row * elements);
    Global destination(reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset);
    pto::Tile<pto::TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT> scratch;
    pto::TASSIGN(scratch, 0);
    auto *workspace = get_dma_workspace(args, DMA_WORKSPACE_SDMA);
    send_request_entry(async_ctx, SdmaTget(destination, source, scratch, workspace));
}
