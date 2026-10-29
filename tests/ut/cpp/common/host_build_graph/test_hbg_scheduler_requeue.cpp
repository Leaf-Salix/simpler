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

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <thread>

#include "scheduler/scheduler_context.h"
#include "callable.h"

namespace {
std::function<void()> register_write_hook;
}  // namespace

// Register writes publish tokens without accessing hardware.
uint64_t __attribute__((weak)) read_reg(uint64_t, RegId) { return 0; }

void __attribute__((weak)) reg_store_release(volatile uint32_t *, uint32_t) {
    if (register_write_hook) register_write_hook();
}

namespace {
ChipReadyQueue *refill_queue = nullptr;
ChipTaskSlotState refill_tasks[2];
bool accept_drain = false;
int drain_calls = 0;
}  // namespace

// Model another producer filling the slots released by dispatch's batch pop.
// This deliberately injects capacity exhaustion; it does not claim the normal
// host sizing contract allows that population. The dispatch and queue operations
// under test are the production implementations.
bool SchedulerContext::enter_drain_mode(ChipTaskSlotState *, int32_t) {
    drain_calls++;
    for (auto &task : refill_tasks) {
        EXPECT_TRUE(refill_queue->push(&task));
    }
    return accept_drain;
}

class SchedulerContextTestPeer {
public:
    static LoopAction check_exit(SchedulerContext &context, SharedMemoryHeader &header) {
        context.total_tasks_ = 2;
        context.completed_tasks_.store(0);
        int32_t count = -1;
        return context.check_exit_conditions(1, &header, nullptr, count);
    }

    static void bind_callable(SchedulerContext &context, uint64_t *table) {
        context.func_id_to_addr_ = table;
        context.func_id_to_addr_count_ = 1;
    }

    static bool dispatch(
        SchedulerContext &context, SchedulerState &scheduler, ChipReadyQueue *queues, bool expected_progress = false
    ) {
        context.sched_ = &scheduler;
        auto &tracker = context.core_trackers_[0];
        tracker.init(2);
        tracker.set_cluster(0, 0, 1, 2);
        tracker.set_cluster(1, 3, 4, 5);
        bool entered_drain = false;
        bool made_progress = false;
        bool try_pushed = false;
        context.dispatch_shape(
            0, queues, ResourceShape::AIC, CoreTracker::DispatchPhase::IDLE, tracker, entered_drain, made_progress,
            try_pushed
        );
        EXPECT_EQ(made_progress, expected_progress);
        EXPECT_EQ(try_pushed, expected_progress);
        return entered_drain;
    }
};

class SchedulerRequeueTest : public ::testing::Test {
protected:
    void SetUp() override {
        scheduler->sm_header = header.get();
        header->sched_error_code.store(SIMPLER_ERROR_NONE);
        header->sched_error_thread.store(-1);
        header->sched_error_bitmap.store(0);
        auto &queue = queues[static_cast<int>(ResourceShape::AIC)];
        queue.capacity = 2;
        queue.mask = 1;
        queue.slots = slots;
        queue.enqueue_pos.store(0);
        queue.dequeue_pos.store(0);
        queue.max_occupancy.store(0);
        queue.seed_slots();
        for (auto &task : tasks) {
            task.task_attrs.set_sync_start();
            task.logical_block_num = 3;
            ASSERT_TRUE(queue.push(&task));
        }
        refill_queue = &queue;
        accept_drain = false;
        drain_calls = 0;
    }

    void expect_overflow() {
        EXPECT_TRUE(SchedulerContextTestPeer::dispatch(*context, *scheduler, queues));
        EXPECT_EQ(drain_calls, 1);
        EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
        EXPECT_EQ(header->sched_error_thread.load(), 0);
        EXPECT_EQ(header->sched_error_bitmap.load(), 1u);
        EXPECT_EQ(SchedulerContextTestPeer::check_exit(*context, *header), LoopAction::BREAK_LOOP);
        EXPECT_TRUE(context->is_completed());
        EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
        EXPECT_EQ(header->sched_error_thread.load(), 0);
        for (auto &task : refill_tasks)
            EXPECT_EQ(refill_queue->pop(), &task);
        EXPECT_EQ(refill_queue->pop(), nullptr);
    }

    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::unique_ptr<SchedulerState> scheduler = std::make_unique<SchedulerState>();
    std::unique_ptr<SharedMemoryHeader> header = std::make_unique<SharedMemoryHeader>();
    ChipReadyQueue queues[3]{};
    ChipReadyQueueSlot slots[2]{};
    ChipTaskSlotState tasks[2]{};
};

TEST_F(SchedulerRequeueTest, RejectedDrainLatchesFailedTaskRequeue) {
    ASSERT_EQ(refill_queue->pop(), &tasks[0]);
    expect_overflow();
}

TEST_F(SchedulerRequeueTest, AcceptedDrainStillLatchesFailedRemainderRequeue) {
    accept_drain = true;
    expect_overflow();
}

TEST_F(SchedulerRequeueTest, RequeueFailurePreservesFirstErrorIdentity) {
    header->sched_error_code.store(SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    header->sched_error_thread.store(7);
    header->sched_error_bitmap.store(1u << 7);
    EXPECT_TRUE(SchedulerContextTestPeer::dispatch(*context, *scheduler, queues));
    EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(header->sched_error_thread.load(), 7);
    EXPECT_EQ(header->sched_error_bitmap.load(), (1u << 7) | 1u);
}

// This register hook is available in the a2a3 implementation. The a5 legacy
// target substitutes write_reg itself, so it cannot inject this interleaving.
TEST_F(SchedulerRequeueTest, ExhaustedCoresReportFailedBatchRequeue) {
    bool hook_available = false;
    register_write_hook = [&]() {
        hook_available = true;
    };
    write_reg(0, RegId::DATA_MAIN_BASE, 0);
    register_write_hook = {};
    if (!hook_available) GTEST_SKIP() << "register-store injection requires a2a3";

    ASSERT_EQ(refill_queue->pop(), &tasks[0]);
    ASSERT_EQ(refill_queue->pop(), &tasks[1]);
    ChipTaskStorage dispatch_tasks[2]{};
    dispatch_tasks[0].slot.logical_block_num = 2;
    dispatch_tasks[1].slot.logical_block_num = 1;
    for (auto &task : dispatch_tasks) {
        task.slot.active_mask = LaneMask::aic();
        ASSERT_TRUE(refill_queue->push(&task.slot));
    }
    CoreCallable callable{};
    callable.set_resolved_addr(0x1000);
    uint64_t table[] = {reinterpret_cast<uint64_t>(&callable)};
    SchedulerContextTestPeer::bind_callable(*context, table);

    std::promise<void> producer_entered;
    auto producer_ready = producer_entered.get_future();
    std::promise<void> dispatch_returned;
    auto dispatch_done = dispatch_returned.get_future();
    bool injected = false;
    register_write_hook = [&]() {
        if (injected) return;
        injected = true;
        for (auto &task : refill_tasks)
            EXPECT_TRUE(refill_queue->push(&task));
        producer_entered.set_value();
    };
    // Always release the injected entries. An implementation that spins forever
    // on full capacity is unblocked, then fails the error assertion below; the
    // test does not mistake timing variability for the correctness failure.
    std::thread release([&]() {
        if (producer_ready.wait_for(std::chrono::seconds(2)) != std::future_status::ready) return;
        dispatch_done.wait_for(std::chrono::milliseconds(100));
        for (auto &task : refill_tasks)
            EXPECT_EQ(refill_queue->pop(), &task);
    });
    EXPECT_FALSE(SchedulerContextTestPeer::dispatch(*context, *scheduler, queues, true));
    dispatch_returned.set_value();
    release.join();
    register_write_hook = {};
    EXPECT_TRUE(injected);
    EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header->sched_error_thread.load(), 0);
    EXPECT_EQ(SchedulerContextTestPeer::check_exit(*context, *header), LoopAction::BREAK_LOOP);
    EXPECT_TRUE(context->is_completed());
}
