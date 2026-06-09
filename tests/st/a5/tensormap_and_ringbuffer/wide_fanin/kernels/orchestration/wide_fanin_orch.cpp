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
        // WideFaninBarrier: K=15 fanin from tensormap only.
        //
        // Constraint: consumer tensor args = 15 INPUT + 1 INOUT(Y) = 16 = MAX_TENSOR_ARGS.
        // Any additional add_input would exceed the limit.
        static constexpr int32_t N = 15;

        Tensor producers[N];
        for (int32_t i = 0; i < N; i++) {
            producers[i] = from_tensor_arg(orch_args.tensor(i));
        }
        Tensor ext_Y = from_tensor_arg(orch_args.tensor(N));

        // Submit N independent producers.
        for (int32_t i = 0; i < N; i++) {
            Arg p_args;
            p_args.add_inout(producers[i]);
            rt_submit_aic_task(FUNC_WRITE_CONST, p_args);
        }
        // Consumer: all N producer tensors as INPUT, Y as INOUT.
        // Each add_input creates one tensormap fanin edge to the corresponding
        // producer.  N=15 lookups -> K=15 fanin from tensormap alone.
        {
            Arg c_args;
            c_args.add_input(producers[0], producers[1], producers[2], producers[3], producers[4], producers[5],
                             producers[6], producers[7], producers[8], producers[9], producers[10], producers[11],
                             producers[12], producers[13], producers[14]);
            c_args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, c_args);
        }
    } else if (case_id == 2) {
        // ExplicitDepWideFanin: K=16 explicit deps + K=1 tensormap = K=17 total.
        //
        // Consumer uses the primitive Arg API's set_dependencies(ptr, count)
        // to bypass ArgWithDeps<16>'s capacity cap.  16 explicit deps on all
        // producer IDs + 1 tensormap edge from add_input(X_0) = 17 fanin.
        static constexpr int32_t N = 16;

        Tensor producers[N];
        for (int32_t i = 0; i < N; i++) {
            producers[i] = from_tensor_arg(orch_args.tensor(i));
        }
        Tensor ext_Y = from_tensor_arg(orch_args.tensor(N));

        // Submit N independent producers and collect their task IDs.
        PTO2TaskId producer_ids[N];
        for (int32_t i = 0; i < N; i++) {
            Arg p_args;
            p_args.add_inout(producers[i]);
            producer_ids[i] = rt_submit_aic_task(FUNC_WRITE_CONST, p_args).task_id();
        }
        // Consumer: 16 explicit deps (primitive Arg, not ArgWithDeps) + 1 input.
        {
            Arg c_args;
            c_args.set_dependencies(producer_ids, N);
            c_args.add_input(producers[0]);
            c_args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, c_args);
        }
    } else {
        rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "unsupported case_id=%llu", static_cast<unsigned long long>(case_id));
    }
}

}  // extern "C"
