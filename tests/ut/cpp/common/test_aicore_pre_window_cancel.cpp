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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "runtime.h"

void aicore_execute(Runtime *runtime, int block_idx, CoreType core_type);

namespace {
std::vector<uint32_t> registers(0x6000 / sizeof(uint32_t));

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

// A subprocess bounds a missing cancellation without leaving a spinning worker
// in the GoogleTest process. The worker runs the production executor TU.
void check_pre_window_cancel(CoreType core_type, bool cancel_before_start) {
    Runtime runtime;
    // With two cube blocks, cube block 1 is worker 1 and its second vector
    // subcore is worker 5 (block * 2 + subcore + cube block count).
    const int worker_index = core_type == CoreType::AIC ? 1 : 5;
    auto &control = runtime.get_teardown_gates()[worker_index].post_close_release;
    for (int generation = 0; generation < 8; ++generation) {
        std::fill(registers.begin(), registers.end(), 0);
        std::memset(runtime.get_workers(), 0, sizeof(Handshake) * RUNTIME_MAX_WORKER);
        __atomic_store_n(
            &runtime.get_teardown_gates()[0].post_close_release, AICORE_PRE_WINDOW_HOST_CANCEL, __ATOMIC_RELEASE
        );
        __atomic_store_n(&control, cancel_before_start ? AICORE_PRE_WINDOW_HOST_CANCEL : 0U, __ATOMIC_RELEASE);
        std::atomic<bool> returned{false};
        std::thread worker([&] {
            aicore_execute(&runtime, worker_index, core_type);
            returned.store(true, std::memory_order_release);
        });
        if (!cancel_before_start) {
            if (!wait_until(
                    [&] {
                        return __atomic_load_n(&runtime.get_workers()[worker_index].aicore_done, __ATOMIC_ACQUIRE) != 0;
                    },
                    std::chrono::seconds(5)
                ))
                std::_Exit(2);
            __atomic_store_n(&control, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELEASE);
            if (wait_until(
                    [&] {
                        return returned.load(std::memory_order_acquire);
                    },
                    std::chrono::milliseconds(10)
                ))
                std::_Exit(3);
            __atomic_store_n(&control, AICORE_PRE_WINDOW_HOST_CANCEL, __ATOMIC_RELEASE);
        }
        if (!wait_until(
                [&] {
                    return returned.load(std::memory_order_acquire);
                },
                std::chrono::seconds(5)
            ))
            std::_Exit(4);
        worker.join();
        if (read_reg(RegId::DATA_MAIN_BASE) != 0 || read_reg(RegId::COND) != 0) std::_Exit(5);
    }
    std::_Exit(0);
}

TEST(AicorePreWindowCancelDeathTest, AivObservesCancelAfterReportingReady) {
    EXPECT_EXIT(check_pre_window_cancel(CoreType::AIV, false), testing::ExitedWithCode(0), "");
}

TEST(AicorePreWindowCancelDeathTest, AicObservesCancelAfterReportingReady) {
    EXPECT_EXIT(check_pre_window_cancel(CoreType::AIC, false), testing::ExitedWithCode(0), "");
}

TEST(AicorePreWindowCancelDeathTest, CancellationBeforeAivStartsIsRetained) {
    EXPECT_EXIT(check_pre_window_cancel(CoreType::AIV, true), testing::ExitedWithCode(0), "");
}

TEST(AicorePreWindowCancelDeathTest, CancellationBeforeAicStartsIsRetained) {
    EXPECT_EXIT(check_pre_window_cancel(CoreType::AIC, true), testing::ExitedWithCode(0), "");
}
}  // namespace

Runtime::Runtime() {
    std::memset(get_workers(), 0, sizeof(Handshake) * RUNTIME_MAX_WORKER);
    std::memset(get_teardown_gates(), 0, sizeof(AicoreTeardownControl) * RUNTIME_MAX_WORKER);
}

volatile uint8_t *sim_get_reg_base() { return reinterpret_cast<volatile uint8_t *>(registers.data()); }
uint32_t sim_get_physical_core_id() { return 0; }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }
struct PmuAicoreRing;
PmuAicoreRing *get_aicore_pmu_ring() { return nullptr; }
uint64_t get_aicore_pmu_reg_base() { return 0; }
