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
 * wide_fanin orchestration cases.
 *
 * Case 1 (WideFaninBarrier): K=15 fanin from tensormap lookup.
 *   N=15 producers write X_i[0]=42.0 (i=0..14).
 *   Consumer: add_input(X_0..X_14) + add_inout(Y) = 16 tensors.
 *   COPY_FIRST reads args[0]=X_0 and writes args[15]=Y.
 *   Y[0]=42.0.  K=15 (one tensormap edge per input tensor).
 *
 * Case 2 (ExplicitDepWideFanin): K=16 explicit deps + 1 tensormap = K=17 total.
 *   N=16 producers write X_i[0]=42.0 (i=0..15).
 *   Consumer: set_dependencies(all_16_ids, 16) + add_input(X_0) + add_inout(Y)
 *           = 2 tensors (X_0 as INPUT, Y as INOUT).
 *   COPY_FIRST reads args[0]=X_0 and writes args[1]=Y.
 *   Y[0]=42.0.  K=16 explicit + 1 tensormap = 17 total.
 *
 * Args layout: [X_0..X_15, Y] (17 tensor slots in callable signature).
 *   Case 1 uses X_0..X_14 + Y (slot 15 = X_15 unused by runtime).
 *   Case 2 uses X_0..X_15 + Y (all 17 slots).
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1
#define FUNC_COPY_FIRST_TO_LAST 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 17 + 1,  // 17 tensors + 1 scalar
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    uint64_t case_id = orch_args.scalar(0);
    LOG_INFO_V0("[wide_fanin_orch] case_id=%llu", static_cast<unsigned long long>(case_id));

    if (case_id == 1) {
        // 15 producers + 1 consumer. Consumer has 15 INPUT + 1 INOUT = 16 tensors.
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
        Tensor ext_Y = from_tensor_arg(orch_args.tensor(16));  // Y at slot 16

        Tensor prods[] = {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14};
        for (int32_t i = 0; i < 15; i++) {
            Arg p_args;
            p_args.add_inout(prods[i]);
            rt_submit_aic_task(FUNC_WRITE_CONST, p_args);
        }
        // Consumer: 15 INPUT + 1 INOUT = 16 tensors (within MAX_TENSOR_ARGS=16)
        {
            Arg c_args;
            c_args.add_input(x0);
            c_args.add_input(x1);
            c_args.add_input(x2);
            c_args.add_input(x3);
            c_args.add_input(x4);
            c_args.add_input(x5);
            c_args.add_input(x6);
            c_args.add_input(x7);
            c_args.add_input(x8);
            c_args.add_input(x9);
            c_args.add_input(x10);
            c_args.add_input(x11);
            c_args.add_input(x12);
            c_args.add_input(x13);
            c_args.add_input(x14);
            c_args.add_inout(ext_Y);
            LOG_INFO_V0("[wide_fanin] case1 consumer tensor_count=%d", c_args.tensor_count());
            rt_submit_aic_task(FUNC_COPY_FIRST_TO_LAST, c_args);
        }
    } else if (case_id == 2) {
        // 16 producers + 1 consumer. Consumer uses explicit deps + 1 INPUT + 1 INOUT = 2 tensors.
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
