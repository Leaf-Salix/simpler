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
 * wide_fanin orchestration cases.
 *
 * Each case is selected via params["case"] in the orchestration scalar slot.
 *
 *   case=1 (WideFaninBarrier): K=15 fanin from tensormap only.
 *     N=15 producers:  producer_i writes X_i[0] = 42.0  (i = 0..14)
 *     1 consumer:      add_input(X_0..X_14) + add_inout(Y)
 *                      -> 15 tensormap lookups -> K=15 fanin edges
 *                      (15 INPUT + 1 INOUT = 16 tensor args, at MAX_TENSOR_ARGS cap)
 *     Consumer copies X_0[0] -> Y[0] via COPY_FIRST.
 *     expect Y[0] = 42.0
 *
 *   case=2 (ExplicitDepWideFanin): K=16 explicit deps + K=1 tensormap = K=17 total.
 *     N=16 producers:  producer_i writes X_i[0] = 42.0  (i = 0..15)
 *     1 consumer:      set_dependencies(all_16_producer_ids, 16)  [primitive Arg API]
 *                      + add_input(X_0) + add_inout(Y)
 *                      -> 16 explicit deps + 1 tensormap edge -> K=17 fanin
 *     Consumer copies X_0[0] -> Y[0] via COPY_FIRST.
 *     expect Y[0] = 42.0
 *
 * Tensor args layout: [X_0, ..., X_{N-1}, Y]
 *   - X_i: produced by producer_i (WRITE_CONST, func_id=0)
 *   - Y: consumer output (COPY_FIRST, func_id=1)
 *
 * Scalar: case selector
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1

static constexpr int32_t MAX_TENSOR_ARGS_LOCAL = 16;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    // case=1: 15 producer tensors + 1 output + 1 scalar = 17
    // case=2: 16 producer tensors + 1 output + 1 scalar = 18
    // Config takes the larger value; both cases use the same tensor slot layout.
    return PTO2OrchestrationConfig{
        .expected_arg_count = MAX_TENSOR_ARGS_LOCAL + 1 + 1,  // 16 + 1 + 1 = 18
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    uint64_t case_id = orch_args.scalar(0);
    LOG_INFO_V0("[wide_fanin_orch] case_id=%llu", static_cast<unsigned long long>(case_id));

    if (case_id == 1) {
        Tensor x0  = from_tensor_arg(orch_args.tensor(0));
        Tensor x1  = from_tensor_arg(orch_args.tensor(1));
        Tensor x2  = from_tensor_arg(orch_args.tensor(2));
        Tensor x3  = from_tensor_arg(orch_args.tensor(3));
        Tensor x4  = from_tensor_arg(orch_args.tensor(4));
        Tensor x5  = from_tensor_arg(orch_args.tensor(5));
        Tensor x6  = from_tensor_arg(orch_args.tensor(6));
        Tensor x7  = from_tensor_arg(orch_args.tensor(7));
        Tensor x8  = from_tensor_arg(orch_args.tensor(8));
        Tensor x9  = from_tensor_arg(orch_args.tensor(9));
        Tensor x10 = from_tensor_arg(orch_args.tensor(10));
        Tensor x11 = from_tensor_arg(orch_args.tensor(11));
        Tensor x12 = from_tensor_arg(orch_args.tensor(12));
        Tensor x13 = from_tensor_arg(orch_args.tensor(13));
        Tensor x14 = from_tensor_arg(orch_args.tensor(14));
        Tensor ext_Y = from_tensor_arg(orch_args.tensor(15));

        // Submit 15 independent producers.
        Tensor prods[] = {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14};
        for (int32_t i = 0; i < 15; i++) {
            Arg p_args;
            p_args.add_inout(prods[i]);
            rt_submit_aic_task(FUNC_WRITE_CONST, p_args);
        }
        // Consumer: all 15 producer tensors as INPUT, Y as INOUT.
        {
            Arg c_args;
            c_args.add_input(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14);
            c_args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, c_args);
        }
    } else if (case_id == 2) {
        Tensor x0  = from_tensor_arg(orch_args.tensor(0));
        Tensor x1  = from_tensor_arg(orch_args.tensor(1));
        Tensor x2  = from_tensor_arg(orch_args.tensor(2));
        Tensor x3  = from_tensor_arg(orch_args.tensor(3));
        Tensor x4  = from_tensor_arg(orch_args.tensor(4));
        Tensor x5  = from_tensor_arg(orch_args.tensor(5));
        Tensor x6  = from_tensor_arg(orch_args.tensor(6));
        Tensor x7  = from_tensor_arg(orch_args.tensor(7));
        Tensor x8  = from_tensor_arg(orch_args.tensor(8));
        Tensor x9  = from_tensor_arg(orch_args.tensor(9));
        Tensor x10 = from_tensor_arg(orch_args.tensor(10));
        Tensor x11 = from_tensor_arg(orch_args.tensor(11));
        Tensor x12 = from_tensor_arg(orch_args.tensor(12));
        Tensor x13 = from_tensor_arg(orch_args.tensor(13));
        Tensor x14 = from_tensor_arg(orch_args.tensor(14));
        Tensor x15 = from_tensor_arg(orch_args.tensor(15));
        Tensor ext_Y = from_tensor_arg(orch_args.tensor(16));

        Tensor prods[] = {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15};
        PTO2TaskId producer_ids[16];
        for (int32_t i = 0; i < 16; i++) {
            Arg p_args;
            p_args.add_inout(prods[i]);
            producer_ids[i] = rt_submit_aic_task(FUNC_WRITE_CONST, p_args).task_id();
        }
        {
            Arg c_args;
            c_args.set_dependencies(producer_ids, 16);
            c_args.add_input(x0);
            c_args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, c_args);
        }
    } else {
        rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "unsupported case_id=%llu", static_cast<unsigned long long>(case_id));
    }
}

}  // extern "C"
