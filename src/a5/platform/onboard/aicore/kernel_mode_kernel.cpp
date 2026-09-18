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
#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/core_type.h"
#include "common/kernel_args.h"
#include "simt_anchor.h"
#include "task_interface/tmr_kernel_context.h"
#include "task_interface/tmr_kernel_control.h"

using simpler::tmr::TmrKernelAicoreArgs;
using simpler::tmr::TmrKernelContextDescriptor;

// This separate ELF carries its own profiling storage rather than importing
// program kernel.cpp, whose ID-0 entry would collide with the kernel-mode one.
// The storage and accessors mirror that file's; only the entry differs. Weak
// linkage coalesces the AIC/AIV definitions, as in the program interface.
[[block_local]] static uint32_t s_aicore_profiling_flag;
// Slot pointer (NOT the dereferenced head address) — see
// aicore_profiling_state.h for the lazy-deref contract.
[[block_local]] static __gm__ uint64_t *s_chip_swimlane_aicore_head_slot;
[[block_local]] static __gm__ ChipSwimlaneActiveHead *s_chip_swimlane_aicore_head;

__attribute__((weak)) __aicore__ void set_aicore_profiling_flag(uint32_t flag) { s_aicore_profiling_flag = flag; }
__attribute__((weak)) __aicore__ uint32_t get_aicore_profiling_flag() { return s_aicore_profiling_flag; }

__attribute__((weak)) __aicore__ void set_chip_swimlane_aicore_head_slot(__gm__ uint64_t *slot_ptr) {
    s_chip_swimlane_aicore_head_slot = slot_ptr;
    s_chip_swimlane_aicore_head = nullptr;  // force lazy resolution on next get
}
__attribute__((weak)) __aicore__ __gm__ ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() {
    // Lazy first-call resolve. AICPU publishes the slot before opening any
    // register window, so it is valid after AICore observes Phase 2 exit.
    if (s_chip_swimlane_aicore_head == nullptr && s_chip_swimlane_aicore_head_slot != nullptr) {
        s_chip_swimlane_aicore_head =
            reinterpret_cast<__gm__ ChipSwimlaneActiveHead *>(*s_chip_swimlane_aicore_head_slot);
    }
    return s_chip_swimlane_aicore_head;
}
__attribute__((weak)) __aicore__ void set_aicore_pmu_ring(__gm__ PmuAicoreRing *) {}
__attribute__((weak)) __aicore__ __gm__ PmuAicoreRing *get_aicore_pmu_ring() { return nullptr; }
__attribute__((weak)) __aicore__ void set_aicore_pmu_reg_base(uint64_t) {}
__attribute__((weak)) __aicore__ uint64_t get_aicore_pmu_reg_base() { return 0; }

extern __aicore__ void aicore_execute_kernel(
    __gm__ Runtime *runtime, __gm__ const TmrKernelContextDescriptor *context, int block_idx, CoreType core_type
);

#ifdef __DAV_VEC__
extern "C" __global__ __aicore__ void aicore_kernel_mode_0_mix_aiv(__gm__ TmrKernelAicoreArgs *args) {
    const int worker = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
    const CoreType type = CoreType::AIV;
#else
extern "C" __global__ __aicore__ void aicore_kernel_mode_0_mix_aic(__gm__ TmrKernelAicoreArgs *args) {
    const int worker = get_block_idx();
    const CoreType type = CoreType::AIC;
#endif
    if (args == nullptr) return;
    dcci(args, SINGLE_CACHE_LINE);
    dsb(static_cast<mem_dsb_t>(0));
    if (args->context_descriptor == 0 || args->context_descriptor % alignof(TmrKernelContextDescriptor) != 0 ||
        args->resident_kernel_args == 0 || args->resident_kernel_args % alignof(KernelArgs) != 0)
        return;
    auto *context = reinterpret_cast<__gm__ const TmrKernelContextDescriptor *>(args->context_descriptor);
    auto *k_args = reinterpret_cast<__gm__ KernelArgs *>(args->resident_kernel_args);
    dcci(k_args, ENTIRE_DATA_CACHE);
    dsb(static_cast<mem_dsb_t>(0));
    if (context->version != simpler::tmr::kTmrKernelContextVersion ||
        context->bytes != sizeof(TmrKernelContextDescriptor) || context->context_generation == 0 ||
        context->self_address != args->context_descriptor ||
        context->resident_kernel_args != args->resident_kernel_args || context->resident_runtime == 0 ||
        context->resident_runtime != reinterpret_cast<uint64_t>(k_args->runtime_args) || context->flags != 0 ||
        context->reserved[0] != 0 || context->reserved[1] != 0 || context->reserved[2] != 0)
        return;

    // The chip swimlane is the one DFX channel a kernel context carries; the
    // others have no kernel-mode path, so their bits never reach the executor.
    const uint32_t swimlane_flag =
        k_args->enable_profiling_flag & static_cast<uint32_t>(SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    set_aicore_profiling_flag(swimlane_flag);
    // The slot CONTENTS are written by AICPU's `chip_swimlane_aicpu_init`,
    // which races with this entry but publishes the slot before opening any
    // register window; the executor dereferences only after Phase 2 exit.
    // Publishing nullptr on a disabled launch keeps a prior launch's freed
    // pointer out of `get_chip_swimlane_aicore_head()`.
    if (swimlane_flag != 0 && k_args->chip_swimlane_aicore_rotation_table != 0) {
        __gm__ uint64_t *head_table = reinterpret_cast<__gm__ uint64_t *>(k_args->chip_swimlane_aicore_rotation_table);
        set_chip_swimlane_aicore_head_slot(&head_table[worker]);
    } else {
        set_chip_swimlane_aicore_head_slot(nullptr);
    }
    set_aicore_pmu_ring(nullptr);
    set_aicore_pmu_reg_base(0);
#ifdef __DAV_VEC__
    // The prepare-owned zero field keeps SIMT metadata in this AIV entry.
    if (k_args->force_simt_anchor) {
        simt_meta_anchor(reinterpret_cast<__gm__ uint32_t *>(k_args));
    }
#endif
    aicore_execute_kernel(k_args->runtime_args, context, worker, type);
}
