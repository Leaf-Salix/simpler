/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * Consumer kernel for wide_fanin case 1: copies args[0][0] -> args[15][0].
 *
 * In wide_fanin case 1 the consumer submits 15 INPUT tensors (X_0..X_14)
 * followed by 1 INOUT tensor (Y).  The runtime packs them as:
 *   args[0]=X_0, args[1]=X_1, ..., args[14]=X_14, args[15]=Y
 *
 * This kernel reads from the first input (X_0) and writes to the last
 * tensor (Y at fixed offset 15).
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

// Y is always the last tensor (slot 127 = N_PRODUCERS_MAX)
static constexpr int32_t Y_IDX = 127;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensor  = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[Y_IDX]);

    __gm__ float *in  = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr)  + in_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    out[0] = in[0];
    dcci(&out[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
