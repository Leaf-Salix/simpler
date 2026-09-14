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
#include "aicpu/kernel_invocation_consumer.h"
#include "aicpu/args_dump_aicpu.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "tensormap_and_ringbuffer/kernel_dispatch.h"

namespace {
void configure_kernel_platform(const KernelArgs &args) {
    set_platform_regs(args.regs);
    set_platform_dump_base(args.dump_data_base);
    set_dump_args_enabled(SIMPLER_GET_DFX_FLAG(args.enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS));
    set_platform_chip_swimlane_base(args.chip_swimlane_data_base);
    set_platform_chip_swimlane_aicore_rotation_table(args.chip_swimlane_aicore_rotation_table);
    set_chip_swimlane_enabled(SIMPLER_GET_DFX_FLAG(args.enable_profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE));
    set_platform_pmu_base(args.pmu_data_base);
    set_platform_pmu_reg_addrs(args.pmu_reg_addrs);
    set_pmu_enabled(SIMPLER_GET_DFX_FLAG(args.enable_profiling_flag, SIMPLER_DFX_FLAG_PMU));
    set_platform_dep_gen_base(args.dep_gen_data_base);
    set_dep_gen_enabled(SIMPLER_GET_DFX_FLAG(args.enable_profiling_flag, SIMPLER_DFX_FLAG_DEP_GEN));
    set_scope_stats_enabled(SIMPLER_GET_DFX_FLAG(args.enable_profiling_flag, SIMPLER_DFX_FLAG_SCOPE_STATS));
    set_platform_scope_stats_base(args.scope_stats_data_base);
    set_platform_phase_base(args.device_wall_data_base);
}
}  // namespace

int consume_kernel_invocation(
    const SimplerKernelDispatchArgs &args, const KernelCallableDeviceResidency &resident, const void *payload,
    size_t payload_bytes
) {
    prepare_kernel_aicpu_thread();
    return simpler::tmr::execute_tmr_kernel_dispatch(args, resident, payload, payload_bytes, configure_kernel_platform);
}
