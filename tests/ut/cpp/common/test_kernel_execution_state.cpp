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

#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <vector>

#include "host/execution_mode_latch.h"
#include "host/kernel_execution_state.h"
#include "host/kernel_context_claim.h"
#include "host/raii_scope_guard.h"

namespace {

/**
 * Fake ops backing the restricted context vocabulary. Handles are small
 * integers cast to pointers; create/destroy counters prove balance, and the
 * failure knobs drive the CLOSING-retry and partial-init paths.
 */
struct FakeContextOps {
    int current_device{3};
    int get_device_rc{0};
    int get_device_calls{0};
    int create_stream_rc_after{-1};  // fail the Nth create_hidden_stream (0-based); -1 = never
    int create_event_rc_after{-1};
    int destroy_failures_remaining{0};

    int streams_created{0};
    int streams_destroyed{0};
    int events_created{0};
    int events_destroyed{0};
    uintptr_t next_handle{1};
    uint32_t requested_event_flag{0};
    std::vector<uint32_t> event_flags_seen;

    static FakeContextOps *self(void *context) { return static_cast<FakeContextOps *>(context); }

    static int get_current_device(void *context, int *device_id) noexcept {
        auto *ops = self(context);
        ++ops->get_device_calls;
        if (ops->get_device_rc != 0) return ops->get_device_rc;
        *device_id = ops->current_device;
        return 0;
    }
    static int create_hidden_stream(void *context, void **stream) noexcept {
        auto *ops = self(context);
        if (ops->create_stream_rc_after >= 0 && ops->streams_created == ops->create_stream_rc_after) return -42;
        ops->streams_created++;
        *stream = reinterpret_cast<void *>(ops->next_handle++);
        return 0;
    }
    static int destroy_hidden_stream(void *context, void *) noexcept {
        auto *ops = self(context);
        if (ops->destroy_failures_remaining > 0) {
            ops->destroy_failures_remaining--;
            return -43;
        }
        ops->streams_destroyed++;
        return 0;
    }
    static int create_event(void *context, uint32_t flag, void **event) noexcept {
        auto *ops = self(context);
        if (ops->create_event_rc_after >= 0 && ops->events_created == ops->create_event_rc_after) return -44;
        ops->events_created++;
        ops->event_flags_seen.push_back(flag);
        *event = reinterpret_cast<void *>(ops->next_handle++);
        return 0;
    }
    static int destroy_event(void *context, void *) noexcept {
        auto *ops = self(context);
        if (ops->destroy_failures_remaining > 0) {
            ops->destroy_failures_remaining--;
            return -43;
        }
        ops->events_destroyed++;
        return 0;
    }

    KernelContextOps table() {
        return KernelContextOps{
            this,          &get_current_device, &create_hidden_stream, &destroy_hidden_stream, requested_event_flag,
            &create_event, &destroy_event
        };
    }
};

constexpr size_t kStreamCount = static_cast<size_t>(KernelStreamKind::Count);
constexpr size_t kEventCount = static_cast<size_t>(KernelEventKind::Count);

TEST(KernelContextClaim, IsolatesRuntimeAndDeviceAndKeepsOwnershipAfterRejectedAcquire) {
    KernelContextClaimRegistry registry;
    KernelContextClaim first, duplicate, other_device, other_runtime;
    ASSERT_EQ(first.acquire(registry, 3, 29), 0);
    EXPECT_EQ(duplicate.acquire(registry, 3, 29), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(other_device.acquire(registry, 4, 29), 0);
    ASSERT_EQ(other_runtime.acquire(registry, 3, 30), 0);
    // A rejected claimant has no authority to release the existing owner.
    duplicate.rollback_initialization();
    EXPECT_EQ(duplicate.acquire(registry, 3, 29), PTO_RUNTIME_ERR_INVALID_STATE);
    first.finish_finalize(0);
    ASSERT_EQ(duplicate.acquire(registry, 3, 29), 0);
    duplicate.finish_finalize(0);
    other_device.finish_finalize(0);
    other_runtime.finish_finalize(0);
}

TEST(KernelContextClaim, InitializationFailureReleasesButCloseFailureRetainsUntilRetry) {
    KernelContextClaimRegistry registry;
    KernelContextClaim first, second;
    FakeContextOps failing;
    failing.create_stream_rc_after = 0;
    KernelExecutionState failed_state;
    {
        ASSERT_EQ(first.acquire(registry, 3, 29), 0);
        auto rollback = RAIIScopeGuard([&]() {
            first.rollback_initialization();
        });
        EXPECT_NE(failed_state.initialize(3, failing.table()), 0);
    }
    EXPECT_FALSE(first.held());
    ASSERT_EQ(second.acquire(registry, 3, 29), 0);
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 1;
    const int failed_close = state.close();
    ASSERT_NE(failed_close, 0);
    second.finish_finalize(failed_close);
    EXPECT_TRUE(second.held());
    EXPECT_EQ(first.acquire(registry, 3, 29), PTO_RUNTIME_ERR_INVALID_STATE);
    const int retried_close = state.close();
    ASSERT_EQ(retried_close, 0);
    second.finish_finalize(retried_close);
    EXPECT_FALSE(second.held());
    ASSERT_EQ(first.acquire(registry, 3, 29), 0);
    first.finish_finalize(0);
}

TEST(KernelContextClaim, DestructionDoesNotReleaseAnUnclosedOwner) {
    KernelContextClaimRegistry registry;
    {
        KernelContextClaim first;
        ASSERT_EQ(first.acquire(registry, 3, 29), 0);
        first.finish_finalize(-43);
    }
    KernelContextClaim second;
    EXPECT_EQ(second.acquire(registry, 3, 29), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelContextClaim, ConcurrentInitializationHasExactlyOneOwner) {
    KernelContextClaimRegistry registry;
    KernelContextClaim first, second;
    std::atomic<bool> start{false};
    int first_status = 0;
    int second_status = 0;
    auto acquire = [&](KernelContextClaim &claim, int &status) {
        while (!start.load(std::memory_order_acquire)) {}
        status = claim.acquire(registry, 3, 29);
    };
    std::thread a([&]() {
        acquire(first, first_status);
    });
    std::thread b([&]() {
        acquire(second, second_status);
    });
    start.store(true, std::memory_order_release);
    a.join();
    b.join();
    EXPECT_NE(first_status == 0, second_status == 0);
    EXPECT_EQ(first_status == 0 ? second_status : first_status, PTO_RUNTIME_ERR_INVALID_STATE);
    first.finish_finalize(0);
    second.finish_finalize(0);
    KernelContextClaim next;
    EXPECT_EQ(next.acquire(registry, 3, 29), 0);
    next.finish_finalize(0);
}

class KernelContextCreationFailure : public ::testing::TestWithParam<int> {};

TEST_P(KernelContextCreationFailure, EveryCreationPointRollsBackAndCanRetry) {
    FakeContextOps fake;
    const int point = GetParam();
    if (point < static_cast<int>(kStreamCount)) {
        fake.create_stream_rc_after = point;
    } else {
        fake.create_event_rc_after = point - kStreamCount;
    }
    KernelExecutionState state;
    EXPECT_NE(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_created, fake.streams_destroyed);
    EXPECT_EQ(fake.events_created, fake.events_destroyed);
    EXPECT_EQ(fake.streams_created + fake.events_created, point);
    fake.create_stream_rc_after = -1;
    fake.create_event_rc_after = -1;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(fake.streams_created, fake.streams_destroyed);
    EXPECT_EQ(fake.events_created, fake.events_destroyed);
}

INSTANTIATE_TEST_SUITE_P(
    AllStreamsAndEvents, KernelContextCreationFailure, ::testing::Range(0, static_cast<int>(kStreamCount + kEventCount))
);

TEST(KernelExecutionState, DestructionWithoutCloseMakesNoRuntimeCalls) {
    FakeContextOps fake;
    {
        KernelExecutionState state;
        ASSERT_EQ(state.initialize(3, fake.table()), 0);
    }
    EXPECT_EQ(fake.streams_destroyed, 0);
    EXPECT_EQ(fake.events_destroyed, 0);
}

TEST(KernelContextOpsVocabulary, IncompleteTableIsInvalid) {
    FakeContextOps fake;
    KernelContextOps ops = fake.table();
    EXPECT_TRUE(ops.valid());
    ops.create_event = nullptr;
    EXPECT_FALSE(ops.valid());
}

TEST(KernelLaunchOpsVocabulary, IncompleteTableIsInvalid) {
    KernelLaunchOps ops{};
    EXPECT_FALSE(ops.valid());
}

TEST(ExecutionModeLatch, StartsUnlatched) {
    ExecutionModeLatch latch;
    EXPECT_FALSE(latch.is_latched());
    EXPECT_FALSE(latch.is_kernel());
}

TEST(ExecutionModeLatch, LatchingIsIdempotentForTheHeldMode) {
    ExecutionModeLatch latch;
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_TRUE(latch.is_latched());
    EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_PROGRAM);
    EXPECT_FALSE(latch.is_kernel());
}

TEST(ExecutionModeLatch, ModesAreMutuallyExclusiveInBothDirections) {
    ExecutionModeLatch program;
    EXPECT_EQ(program.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(program.latch(SIMPLER_MODE_KERNEL), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(program.is_kernel());

    ExecutionModeLatch kernel;
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_KERNEL), 0);
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_KERNEL), 0);
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_PROGRAM), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(kernel.is_kernel());
}

// finalize resets a runner's device state but not its identity, so the
// init -> finalize -> init sequence the program path already supports must
// still latch cleanly the second time.
TEST(ExecutionModeLatch, SameModeRelatchesAfterAFinalizedLifetime) {
    ExecutionModeLatch latch;
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_PROGRAM);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_KERNEL), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelExecutionState, EmptyContextInitThenCloseIsClean) {
    FakeContextOps fake;
    KernelExecutionState state;
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Collecting);
    EXPECT_TRUE(state.accepts_dispatch());
    EXPECT_EQ(state.device_id(), 3);
    EXPECT_EQ(fake.streams_created, static_cast<int>(kStreamCount));
    EXPECT_EQ(fake.events_created, static_cast<int>(kEventCount));
    for (size_t i = 0; i < kStreamCount; ++i) {
        EXPECT_NE(state.hidden_stream(static_cast<KernelStreamKind>(i)), nullptr);
    }
    for (size_t i = 0; i < kEventCount; ++i) {
        EXPECT_NE(state.event(static_cast<KernelEventKind>(i)), nullptr);
    }

    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_destroyed, fake.events_created);
    // Idempotent close.
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionState, EveryEventCarriesTheFlagTheOpsTableAsksFor) {
    constexpr uint32_t kPlatformEventFlag = 0x8;
    FakeContextOps fake;
    fake.requested_event_flag = kPlatformEventFlag;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);

    ASSERT_EQ(fake.event_flags_seen.size(), kEventCount);
    for (uint32_t flag : fake.event_flags_seen) {
        EXPECT_EQ(flag, kPlatformEventFlag);
    }
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionState, CloseOnNewContextIsANoOpSuccess) {
    KernelExecutionState state;
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
}

TEST(KernelExecutionState, RepeatedInitializeRejected) {
    FakeContextOps fake;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.phase(), KernelContextPhase::Collecting);
}

TEST(KernelExecutionState, PreEnqueueValidationLeavesStateUnchanged) {
    FakeContextOps fake;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(-1, fake.table()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    KernelContextOps incomplete = fake.table();
    incomplete.destroy_event = nullptr;
    EXPECT_EQ(state.initialize(3, incomplete), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(fake.get_device_calls, 0);
    EXPECT_EQ(fake.streams_created, 0);
    EXPECT_EQ(fake.events_created, 0);
    // The context stays usable after the clean rejections.
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(fake.get_device_calls, 1);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_destroyed, fake.events_created);
}

TEST(KernelExecutionState, DeviceMismatchRejectedBeforeAnyCreation) {
    FakeContextOps fake;
    fake.current_device = 7;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(fake.streams_created, 0);
    EXPECT_EQ(fake.events_created, 0);
}

TEST(KernelExecutionState, PartialInitFailureRollsBackCleanly) {
    FakeContextOps fake;
    fake.create_event_rc_after = 1;  // second event creation fails
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), -44);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_destroyed, fake.events_created);
    // The rolled-back context is reusable.
    fake.create_event_rc_after = -1;
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
}

TEST(KernelExecutionState, StreamCreateFailureRollsBackCleanly) {
    FakeContextOps fake;
    fake.create_stream_rc_after = 1;  // second hidden-stream creation fails
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), -42);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_created, 0);
}

TEST(KernelExecutionState, GetCurrentDeviceFailurePropagates) {
    FakeContextOps fake;
    fake.get_device_rc = -51;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), -51);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(fake.streams_created, 0);
    EXPECT_EQ(fake.events_created, 0);
}

TEST(KernelExecutionState, InitFailureWithFailedCleanupLandsInClosing) {
    FakeContextOps fake;
    fake.create_event_rc_after = 0;       // first event creation fails
    fake.destroy_failures_remaining = 1;  // the rollback's first destroy also fails
    KernelExecutionState state;
    // The reported error is the create failure; the cleanup failure is
    // latched in the separate teardown slot.
    EXPECT_EQ(state.initialize(3, fake.table()), -44);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(state.unexpected_teardown_error(), -43);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_FALSE(state.accepts_dispatch());
    // The kept ops table lets an explicit close retry release the remainder.
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(state.has_live_resources());
}

TEST(KernelExecutionState, ReadyEnqueuedKeepsAdmissionOpen) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.mark_ready_enqueued(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::ReadyEnqueued);
    EXPECT_EQ(state.mark_ready_enqueued(), 0);  // append admission stays open
    EXPECT_TRUE(state.accepts_dispatch());
}

TEST(KernelExecutionState, MarkReadyEnqueuedRejectedOffPath) {
    KernelExecutionState state;
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelExecutionState, PoisonRejectsDispatchButAllowsClose) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    state.poison(-7);
    EXPECT_EQ(state.phase(), KernelContextPhase::Poisoned);
    EXPECT_FALSE(state.accepts_dispatch());
    EXPECT_EQ(state.last_runtime_error(), -7);
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    // First poison cause wins; later ones do not overwrite it.
    state.poison(-8);
    EXPECT_EQ(state.last_runtime_error(), -7);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
}

TEST(KernelExecutionState, PoisonOutsideDispatchPhasesIsIgnored) {
    KernelExecutionState state;
    state.poison(-9);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(state.last_runtime_error(), 0);
}

TEST(KernelExecutionState, PoisonDuringClosingIsIgnored) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 1;
    ASSERT_EQ(state.close(), -43);
    ASSERT_EQ(state.phase(), KernelContextPhase::Closing);
    state.poison(-9);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(state.last_runtime_error(), 0);
}

TEST(KernelExecutionState, PoisonAfterCloseIsIgnored) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    ASSERT_EQ(state.close(), 0);
    state.poison(-9);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_EQ(state.last_runtime_error(), 0);
}

TEST(KernelExecutionState, InitializeRejectedFromEveryNonNewPhase) {
    FakeContextOps fake;

    KernelExecutionState ready;
    ASSERT_EQ(ready.initialize(3, fake.table()), 0);
    ASSERT_EQ(ready.mark_ready_enqueued(), 0);
    EXPECT_EQ(ready.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(ready.phase(), KernelContextPhase::ReadyEnqueued);

    KernelExecutionState poisoned;
    ASSERT_EQ(poisoned.initialize(3, fake.table()), 0);
    poisoned.poison(-7);
    EXPECT_EQ(poisoned.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(poisoned.phase(), KernelContextPhase::Poisoned);

    KernelExecutionState closing;
    ASSERT_EQ(closing.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 1;
    ASSERT_EQ(closing.close(), -43);
    ASSERT_EQ(closing.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(closing.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(closing.phase(), KernelContextPhase::Closing);

    KernelExecutionState closed;
    EXPECT_EQ(closed.close(), 0);
    EXPECT_EQ(closed.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(closed.phase(), KernelContextPhase::Closed);
}

TEST(KernelExecutionState, ClosingIsStickyAndRetriable) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 2;
    const int rc = state.close();
    EXPECT_EQ(rc, -43);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_FALSE(state.accepts_dispatch());
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_EQ(state.unexpected_teardown_error(), -43);

    // Retry succeeds and releases only what is still live.
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_destroyed, fake.events_created);
}

TEST(KernelExecutionState, TeardownErrorSlotIsSeparateFromRuntimeError) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    state.poison(-7);
    fake.destroy_failures_remaining = 1;
    EXPECT_EQ(state.close(), -43);
    // The controlled poison cause and the real teardown failure are reported
    // through separate slots; neither masks the other.
    EXPECT_EQ(state.last_runtime_error(), -7);
    EXPECT_EQ(state.unexpected_teardown_error(), -43);
    EXPECT_EQ(state.close(), 0);
    // The first teardown failure stays latched after a successful retry.
    EXPECT_EQ(state.unexpected_teardown_error(), -43);
}

struct FakeResourceOps {
    int allocations{0};
    int releases{0};
    int release_failures{0};

    KernelResourceOps table() {
        return {
            this,
            [](void *context, size_t bytes) -> void * {
                ++static_cast<FakeResourceOps *>(context)->allocations;
                return std::malloc(bytes);
            },
            [](void *context, void *ptr) {
                auto &fake = *static_cast<FakeResourceOps *>(context);
                if (fake.release_failures > 0) {
                    --fake.release_failures;
                    return -71;
                }
                ++fake.releases;
                std::free(ptr);
                return 0;
            },
        };
    }
};

KernelResourceLayout resource_layout() {
    KernelResourceLayout layout;
    layout.schema = 41;
    layout.contract.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
    layout.contract.pipeline_depth = 1;
    layout.contract.resource_count = 3;
    layout.contract.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DEVICE_SCRATCH, 4096};
    layout.contract.resources[1] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    layout.contract.resources[2] = {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    layout.regions = {{PTO_PIPELINE_GM_HEAP, 0, 4096}};
    return layout;
}

TEST(KernelExecutionStateResources, ZeroGenerationIsRejectedBeforeCreatingHandles) {
    FakeContextOps handles;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, handles.table(), 0), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(handles.streams_created, 0);
    EXPECT_EQ(handles.events_created, 0);
}

TEST(KernelExecutionStateResources, FrozenBindingChecksGenerationAndPreservesAddressAfterRefusal) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    ASSERT_TRUE(state.resources_prepared());
    ASSERT_EQ(state.freeze_resources(), 0);
    ASSERT_TRUE(state.resources_frozen());
    const uint64_t required[] = {4096};
    KernelResourceBinding binding;
    EXPECT_EQ(state.bind_resources_for_launch(3, 19, 41, required, 1, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(state.inspect_frozen_resources(3, 19, 41, required, 1, binding), 0);
    const auto address = binding.regions[0].address;
    ASSERT_NE(address, 0u);
    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    EXPECT_EQ(state.bind_resources_for_launch(3, 18, 41, required, 1, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.bind_resources_for_launch(4, 19, 41, required, 1, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    const uint64_t too_large[] = {4097};
    EXPECT_EQ(state.bind_resources_for_launch(3, 19, 41, too_large, 1, binding), PTO_RUNTIME_ERR_CAPACITY_EXCEEDED);
    ASSERT_EQ(state.bind_resources_for_launch(3, 19, 41, required, 1, binding), 0);
    EXPECT_EQ(binding.regions[0].address, address);
    EXPECT_EQ(memory.allocations, 1);
    EXPECT_EQ(memory.releases, 0);
    ASSERT_EQ(state.close(), 0);
    EXPECT_EQ(memory.releases, 1);
}

TEST(KernelExecutionStateResources, MemoryCloseFailureKeepsHandlesForRetry) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    memory.release_failures = 1;
    EXPECT_EQ(state.close(), -71);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_EQ(state.unexpected_teardown_error(), -71);
    EXPECT_EQ(handles.streams_destroyed, 0);
    EXPECT_EQ(handles.events_destroyed, 0);
    EXPECT_EQ(state.prepare_resources(resource_layout(), memory.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(state.close(), 0);
    EXPECT_EQ(memory.releases, 1);
    EXPECT_EQ(handles.streams_destroyed, handles.streams_created);
    EXPECT_EQ(handles.events_destroyed, handles.events_created);
    EXPECT_FALSE(state.has_live_resources());
}

TEST(KernelExecutionStateResources, PreparationChecksTheBorrowedCurrentDeviceBeforeAllocating) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    handles.current_device = 4;
    EXPECT_EQ(state.prepare_resources(resource_layout(), memory.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(memory.allocations, 0);
    handles.current_device = 3;
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    EXPECT_EQ(memory.allocations, 1);
    EXPECT_EQ(state.close(), 0);
}

}  // namespace
