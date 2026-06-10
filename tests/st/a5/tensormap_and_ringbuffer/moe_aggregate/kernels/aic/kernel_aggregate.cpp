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

/**
 * AGGREGATE kernel: sums k expert tensors into Y.
 *
 * Kernel args layout (variable-position, determined by k):
 *   args[0..k-1] = expert tensor pointers (INPUT)
 *   args[k]      = Y tensor pointer (INOUT)
 *   args[k+1]    = mode_and_k scalar (uint64_t):
 *                    low 16 bits  = k (number of experts to sum)
 *                    bit  16      = mode (0 = Y[0] = sum, 1 = Y[0] += sum)
 *
 * The zero_task (WRITE_CONST) writes 42.0f to Y[0] as a WAR barrier.
 * In mode 0, this kernel overwrites Y[0] (the zero_task value is lost).
 * In mode 1, this kernel adds to Y[0] (preserving prior aggregation).
 *
 * For K=16 (absolute max fanin), the orchestration calls AGGREGATE twice:
 *   call 1 (mode 0): k=15 experts[0..14], Y[0] = sum = 630.0
 *   call 2 (mode 1): k=1  experts[15],     Y[0] += 42.0 = 672.0
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    // The mode_and_k scalar is at args[k+1].  We find it by scanning from
    // args[2] upward for the first non-64-aligned value (tensor pointers are
    // 64-byte aligned; the scalar k is 1..16 and bit 16 may be set for mode).
    uint64_t mode_and_k = 0;
    uint64_t y_idx = 0;
    for (uint64_t i = 1; i <= 16; i++) {
        uint64_t val = static_cast<uint64_t>(args[i]);
        // Tensor pointers are 64-byte aligned (bottom 6 bits = 0).
        // Scalar mode_and_k: low 16 bits = k (1..16), bit 16 = mode (0 or 1).
        // For k >= 1, bottom 6 bits are non-zero; for mode=1, bit 16 is set.
        if ((val & 63) != 0) {
            mode_and_k = val;
            y_idx = i - 1;
            break;
        }
    }

    uint64_t k = mode_and_k & 0xFFFF;
    uint64_t mode = (mode_and_k >> 16) & 1;

    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[y_idx]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    float sum = 0.0f;
    for (uint64_t i = 0; i < k; i++) {
        __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[i]);
        __gm__ float *in = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
        sum += in[0];
    }

    if (mode == 0) {
        out[0] = sum;
    } else {
        out[0] += sum;
    }
    dcci(&out[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
