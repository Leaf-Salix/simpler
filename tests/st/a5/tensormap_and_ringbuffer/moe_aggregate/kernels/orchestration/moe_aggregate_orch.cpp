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
 * MoE aggregation orchestration scenes.
 *
 * Tensor args layout: [experts[0], ..., experts[15], Y]
 *   - experts[i]: produced by producer_i (WRITE_CONST, func_id=0)
 *   - Y: consumer output / aggregation target
 *
 * Scalar: K (number of experts to aggregate, 1..16)
 *
 * K=1 (CopyFirst):
 *   1 producer writes experts[0][0] = 42.0.
 *   Consumer copies experts[0][0] -> Y[0] via COPY_FIRST.
 *   expect Y[0] = 42.0
 *
 * K=2..15 (Aggregate, pure tensormap):
 *   K producers each write experts[i][0] = 42.0.
 *   Zero task writes Y[0] = 42.0 (WAR barrier via INOUT tensormap dep).
 *   Consumer aggregates: Y[0] = sum(experts[0..K-1][0]) via AGGREGATE.
 *   Tensor arg budget: K INPUT + 1 INOUT (at most 15 + 1 = 16 = MAX_TENSOR_ARGS).
 *   expect Y[0] = K * 42.0
 *
 * K=16 (Aggregate, two-phase):
 *   16 producers each write experts[i][0] = 42.0.
 *   Zero task writes Y[0] (WAR barrier).
 *   Phase 1: AGGREGATE(mode=0) sums experts[0..14] -> Y[0] = 630.0
 *     15 INPUT + 1 INOUT = 16 tensor args (at MAX_TENSOR_ARGS cap).
 *     explicit deps on producers[0..14] for scheduling.
 *   Phase 2: AGGREGATE(mode=1) adds experts[15] -> Y[0] += 42.0 = 672.0
 *     1 INPUT (experts[15]) + 1 INOUT (Y) = 2 tensor args.
 *     tensormap dep on Phase 1 (via add_inout(Y)) + explicit dep on producer[15].
 *   expect Y[0] = 16 * 42.0 = 672.0
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_CONST 0
#define FUNC_COPY_FIRST 1
#define FUNC_AGGREGATE 2

static constexpr int32_t MAX_EXPERTS = 16;

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = MAX_EXPERTS + 1 + 1,  // 16 expert tensors + Y + K scalar = 18
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    uint64_t k = orch_args.scalar(0);
    LOG_INFO_V0("[moe_aggregate_orch] K=%llu", static_cast<unsigned long long>(k));

    Tensor experts[MAX_EXPERTS];
    for (int32_t i = 0; i < MAX_EXPERTS; i++) {
        experts[i] = from_tensor_arg(orch_args.tensor(i));
    }
    Tensor ext_Y = from_tensor_arg(orch_args.tensor(MAX_EXPERTS));

    PTO2TaskId producer_ids[MAX_EXPERTS];

    // Submit K independent producers: each writes 42.0f to experts[i][0].
    for (uint64_t i = 0; i < k; i++) {
        Arg p_args;
        p_args.add_inout(experts[i]);
        producer_ids[i] = rt_submit_aic_task(FUNC_WRITE_CONST, p_args).task_id();
    }

    if (k == 1) {
        // K=1: direct copy, no aggregation needed.
        // Consumer args: [experts[0] INPUT, Y INOUT]
        //   args[0] = experts[0], args[1] = Y
        Arg c_args;
        c_args.add_input(experts[0]);
        c_args.add_inout(ext_Y);
        rt_submit_aic_task(FUNC_COPY_FIRST, c_args);
    } else {
        // K>1: zero Y (WAR barrier), then aggregate K experts into Y.
        {
            Arg z_args;
            z_args.add_inout(ext_Y);
            rt_submit_aic_task(FUNC_WRITE_CONST, z_args);
            // WRITE_CONST writes 42.0f to Y[0]. AGGREGATE(mode=0) overwrites
            // Y[0] with sum, so the 42.0 is lost. The zero_task is submitted
            // so the consumer's add_inout(ext_Y) creates a tensormap dep,
            // ensuring Y is not read while being written.
        }

        if (k <= 15) {
            // K=2..15: single AGGREGATE call.
            // Consumer args: [experts[0..k-1] INPUT, Y INOUT, mode_and_k SCALAR]
            //   args[0..k-1] = experts, args[k] = Y, args[k+1] = mode_and_k
            // Tensor arg budget: k INPUT + 1 INOUT (at most 15 + 1 = 16).
            Arg c_args;
            for (uint64_t i = 0; i < k; i++) {
                c_args.add_input(experts[i]);
            }
            c_args.add_inout(ext_Y);
            c_args.add_scalar(k);  // mode=0 (bit 16 = 0), k = low 16 bits
            rt_submit_aic_task(FUNC_AGGREGATE, c_args);
        } else {
            // K=16: two-phase aggregation (16 experts exceed MAX_TENSOR_ARGS).
            // Phase 1: AGGREGATE(mode=0) sums experts[0..14] -> Y[0] = 630.0
            //   15 INPUT + 1 INOUT = 16 tensor args (at cap).
            //   explicit deps on producers[0..14] for scheduling.
            PTO2TaskId phase1_id;
            {
                Arg c_args;
                for (int32_t i = 0; i < 15; i++) {
                    c_args.add_input(experts[i]);
                }
                c_args.add_inout(ext_Y);
                c_args.add_scalar(static_cast<uint64_t>(15));  // mode=0, k=15
                c_args.set_dependencies(producer_ids, 15);
                phase1_id = rt_submit_aic_task(FUNC_AGGREGATE, c_args).task_id();
            }
            // Phase 2: AGGREGATE(mode=1) adds experts[15] -> Y[0] += 42.0 = 672.0
            //   1 INPUT (experts[15]) + 1 INOUT (Y) = 2 tensor args.
            //   tensormap dep on Phase 1 (via add_inout(Y)).
            //   explicit dep on producer[15] (experts[15] must be ready).
            {
                Arg c_args;
                c_args.add_input(experts[15]);
                c_args.add_inout(ext_Y);
                c_args.add_scalar(static_cast<uint64_t>((1 << 16) | 1));  // mode=1, k=1
                PTO2TaskId deps[] = {producer_ids[15]};
                c_args.set_dependencies(deps, 1);
                rt_submit_aic_task(FUNC_AGGREGATE, c_args);
            }
        }
    }
}

}  // extern "C"
