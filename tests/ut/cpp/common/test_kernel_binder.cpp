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

#include <string>
#include <vector>

#include "host/kernel_binder.h"

namespace {
struct OrderState {
    std::vector<std::string> order;
};
int record_step(void *context, const char *name) noexcept {
    static_cast<OrderState *>(context)->order.emplace_back(name);
    return 0;
}
#define DEFINE_STEP(name) \
    int name(void *context) noexcept { return record_step(context, #name); }
DEFINE_STEP(clear_round)
DEFINE_STEP(record_start)
DEFINE_STEP(wait_aicore_start)
DEFINE_STEP(launch_aicore)
DEFINE_STEP(record_aicore_done)
DEFINE_STEP(wait_aicpu_start)
DEFINE_STEP(launch_aicpu)
DEFINE_STEP(record_aicpu_done)
DEFINE_STEP(join_aicpu)
DEFINE_STEP(join_aicore)
DEFINE_STEP(record_tail)
#undef DEFINE_STEP
}  // namespace

TEST(KernelBinderTest, EnforcesForkJoinOrder) {
    OrderState state;
    KernelBinder::Operations ops;
    ops.context = &state;
    ops.clear_round = clear_round;
    ops.record_start = record_start;
    ops.wait_aicore_start = wait_aicore_start;
    ops.launch_aicore = launch_aicore;
    ops.record_aicore_done = record_aicore_done;
    ops.wait_aicpu_start = wait_aicpu_start;
    ops.launch_aicpu = launch_aicpu;
    ops.record_aicpu_done = record_aicpu_done;
    ops.join_aicpu = join_aicpu;
    ops.join_aicore = join_aicore;
    ops.record_tail = record_tail;

    ASSERT_EQ(KernelBinder().submit(ops), 0);
    EXPECT_EQ(
        state.order,
        std::vector<std::string>(
            {"clear_round", "record_start", "wait_aicore_start", "launch_aicore", "record_aicore_done",
             "wait_aicpu_start", "launch_aicpu", "record_aicpu_done", "join_aicpu", "join_aicore", "record_tail"}
        )
    );
}

TEST(KernelBinderTest, FailurePoisonsAndInvokesCancellation) {
    struct State {
        int poisoned{0};
        int cancelled{0};
        int calls{0};
    } state;
    KernelBinder::Operations ops;
    ops.context = &state;
    ops.clear_round = [](void *context) noexcept {
        ++static_cast<State *>(context)->calls;
        return 0;
    };
    ops.record_start = [](void *context) noexcept {
        ++static_cast<State *>(context)->calls;
        return -7;
    };
    ops.poison = [](void *context, int error) noexcept {
        static_cast<State *>(context)->poisoned = error;
    };
    ops.cancel = [](void *context, int error) noexcept {
        static_cast<State *>(context)->cancelled = error;
    };

    EXPECT_EQ(KernelBinder().submit(ops), -7);
    EXPECT_EQ(state.calls, 2);
    EXPECT_EQ(state.poisoned, -7);
    EXPECT_EQ(state.cancelled, -7);
}
