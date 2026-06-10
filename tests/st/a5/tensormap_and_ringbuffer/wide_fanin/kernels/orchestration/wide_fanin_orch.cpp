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
 * wide_fanin orchestration — stress fanin dedup via explicit deps at various K values.
 *
 * Scalar args: case_id (scalar 0), n_producers (scalar 1).
 * Tensor layout: x_0..x_126, y  (128 tensors, Y always at slot 127).
 *
 * Case 2 (ExplicitDepK17):  N=16 producers, consumer explicit deps(N) + 1 tensormap → K=N+1.
 * Case 3 (WideFaninK64/128): N=64/127 producers, consumer explicit deps(N) + 1 tensormap → K=N+1.
 */

#include <cstdint>
#include <vector>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1
#define FUNC_COPY_FIRST_TO_LAST 2

static constexpr int32_t Y_SLOT = 127;  // Y is always at tensor slot 127

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 128 + 2,  // 128 tensors + 2 scalars
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    uint64_t case_id = orch_args.scalar(0);
    int32_t n = static_cast<int32_t>(orch_args.scalar(1));
    LOG_INFO_V0("[wide_fanin] case=%llu n=%d", static_cast<unsigned long long>(case_id), n);

    Tensor ext_Y = from_tensor_arg(orch_args.tensor(Y_SLOT));

    if (case_id == 2 || case_id == 3) {
        // K=N+1 via explicit deps: N producers, consumer explicit deps(N) + 1 tensormap.
        std::vector<Tensor> prods;
        prods.reserve(n);
        for (int32_t i = 0; i < n; i++) {
            prods.push_back(from_tensor_arg(orch_args.tensor(i)));
        }
        // Use heap-allocated PTO2TaskId array for large N.
        std::vector<PTO2TaskId> producer_ids(n);
        for (int32_t i = 0; i < n; i++) {
            Arg p_args;
            p_args.add_inout(prods[i]);
            producer_ids[i] = rt_submit_aic_task(FUNC_WRITE_CONST, p_args).task_id();
        }
        {
            Arg c_args;
            c_args.set_dependencies(producer_ids.data(), n);
            c_args.add_input(prods[0]);
            c_args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_COPY_FIRST, c_args);
        }
    } else {
        rt_report_fatal(PTO2_ERROR_INVALID_ARGS, "unsupported case_id=%llu", static_cast<unsigned long long>(case_id));
    }
}

}  // extern "C"
