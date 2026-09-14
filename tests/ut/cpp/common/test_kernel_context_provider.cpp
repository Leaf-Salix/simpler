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

#include <gtest/gtest.h>

#include "host/kernel_context_provider.h"

TEST(KernelContextProviderTest, PublishesStableViewsAndCloses) {
    KernelContextProvider provider;
    KernelStableDeviceBinding binding{};
    binding.context_generation = 7;
    binding.runtime_address = 0x1000;
    binding.sm_address = 0x2000;
    binding.sm_capacity = 0x1000;
    binding.arena_address = 0x4000;
    binding.arena_capacity = 0x2000;
    binding.runtime_offset = 0x100;
    binding.aicpu_thread_count = 2;
    ASSERT_EQ(provider.prepare_borrowed_binding(binding), 0);
    ASSERT_EQ(provider.freeze_borrowed_binding(), 0);
    ASSERT_NE(provider.binding_view(), nullptr);

    simpler::tmr::TmrKernelContextDescriptor descriptor{};
    descriptor.context_generation = binding.context_generation;
    descriptor.self_address = 0x5000;
    descriptor.resident_runtime = 0x6000;
    descriptor.resident_kernel_args = 0x7000;
    descriptor.sm_base = binding.sm_address;
    descriptor.sm_capacity = binding.sm_capacity;
    descriptor.arena_base = binding.arena_address;
    descriptor.arena_capacity = binding.arena_capacity;
    descriptor.runtime_offset = binding.runtime_offset;
    descriptor.control_address = 0x8000;
    descriptor.control_bytes = sizeof(simpler::tmr::TmrLaunchControl);
    descriptor.reports_address = 0x9000;
    descriptor.reports_bytes = sizeof(simpler::tmr::TmrCoreReport);
    ASSERT_EQ(provider.publish_context_descriptor(descriptor), 0);
    ASSERT_NE(provider.descriptor_view(), nullptr);

    simpler::tmr::TmrKernelClearBinding clear{};
    clear.context_generation = binding.context_generation;
    clear.control = {0x8000, sizeof(simpler::tmr::TmrLaunchControl)};
    clear.reports = {0x9000, sizeof(simpler::tmr::TmrCoreReport)};
    clear.worker_count = 1;
    simpler::tmr::TmrKernelClearPlan plan;
    ASSERT_TRUE(simpler::tmr::build_tmr_kernel_clear_plan(clear, &plan));
    ASSERT_EQ(provider.publish_clear_plan(plan), 0);
    EXPECT_NE(provider.clear_plan_view(), nullptr);
    EXPECT_TRUE(provider.close());
    EXPECT_EQ(provider.binding_view(), nullptr);
    EXPECT_EQ(provider.clear_plan_view(), nullptr);
    EXPECT_EQ(provider.descriptor_view(), nullptr);
}

TEST(KernelContextProviderTest, ResourceLifecycleIsOwnedByProvider) {
    KernelContextProvider provider;
    struct Counts {
        int allocs{0};
        int frees{0};
    } counts;
    KernelResourceOps ops{
        &counts,
        [](void *ctx, size_t bytes) -> void * {
            ++static_cast<Counts *>(ctx)->allocs;
            return ::operator new(bytes);
        },
        [](void *ctx, void *ptr) -> int {
            ++static_cast<Counts *>(ctx)->frees;
            ::operator delete(ptr);
            return 0;
        }
    };
    KernelResourceLayout layout;
    layout.schema = 1;
    layout.contract.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
    layout.contract.resource_count = 6;
    layout.contract.pipeline_depth = 1;
    layout.contract.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DEVICE_SCRATCH, 64};
    layout.contract.resources[1] = {PTO_PIPELINE_GM_SM, PTO_PIPELINE_DEVICE_SCRATCH, 64};
    layout.contract.resources[2] = {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DEVICE_SCRATCH, 64};
    layout.contract.resources[3] = {PTO_PIPELINE_TASK_ARGS, PTO_PIPELINE_HOST_PER_RUN, 0};
    layout.contract.resources[4] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    layout.contract.resources[5] = {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    layout.regions = {{PTO_PIPELINE_GM_SM, 0, 64}};
    ASSERT_EQ(provider.prepare(layout, ops), 0);
    ASSERT_EQ(provider.freeze(), 0);
    KernelStableDeviceBinding binding{};
    binding.context_generation = 1;
    binding.runtime_address = 0x1000;
    binding.sm_address = 0x2000;
    binding.sm_capacity = 64;
    binding.arena_address = 0x4000;
    binding.arena_capacity = 64;
    binding.aicpu_thread_count = 2;
    ASSERT_EQ(provider.prepare_borrowed_binding(binding), 0);
    ASSERT_EQ(provider.freeze_borrowed_binding(), 0);
    KernelResourceBinding view{};
    const uint64_t required[] = {64};
    ASSERT_EQ(provider.borrow(1, required, 1, view), 0);
    EXPECT_EQ(view.count, 1U);
    EXPECT_TRUE(provider.close());
    EXPECT_EQ(counts.allocs, counts.frees);
}

TEST(KernelContextProviderTest, ReleaseExceptionKeepsStorageForRetry) {
    KernelContextProvider provider;
    struct State {
        bool throw_once{true};
        int releases{0};
    } state;
    KernelResourceOps ops{
        &state,
        [](void *, size_t bytes) -> void * {
            return ::operator new(bytes);
        },
        [](void *ctx, void *ptr) -> int {
            auto *state = static_cast<State *>(ctx);
            ++state->releases;
            if (state->throw_once) {
                state->throw_once = false;
                throw 1;
            }
            ::operator delete(ptr);
            return 0;
        }
    };
    KernelStableDeviceBinding binding{};
    binding.context_generation = 1;
    binding.runtime_address = 0x1000;
    binding.sm_address = 0x2000;
    binding.sm_capacity = 64;
    binding.arena_address = 0x4000;
    binding.arena_capacity = 64;
    binding.aicpu_thread_count = 2;
    ASSERT_EQ(provider.prepare_borrowed_binding(binding), 0);
    ASSERT_EQ(provider.freeze_borrowed_binding(), 0);
    void *descriptor = ops.allocate(ops.context, 16);
    void *table = ops.allocate(ops.context, 16);
    void *control = ops.allocate(ops.context, 16);
    void *reports = ops.allocate(ops.context, 16);
    ASSERT_EQ(provider.adopt_tmr_storage(ops, descriptor, table, control, reports), 0);

    EXPECT_FALSE(provider.close());
    EXPECT_TRUE(provider.owns_tmr_storage());
    EXPECT_TRUE(provider.close());
    EXPECT_FALSE(provider.owns_tmr_storage());
    EXPECT_EQ(state.releases, 5);
}
