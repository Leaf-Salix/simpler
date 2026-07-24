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
 * Unit tests for PTO2TaskAllocator from pto_ring_buffer.h
 *
 * Tests ring buffer allocation, heap bump logic, wrap-around, alignment,
 * task window flow control, and heap_available semantics.
 *
 * The allocator has one caller (the orchestrator thread), while scheduler
 * threads concurrently advance its reclaim watermark and error channel.
 *
 * Design contracts (try_bump_heap):
 *
 * - Wrap-around guard uses `tail > alloc_size` (strict >).  When
 *   tail == alloc_size the wrap branch returns nullptr.  Allowing it
 *   would create top == tail (full/empty ambiguity).  Strict >
 *   sacrifices one quantum of capacity.
 *
 * - heap_available() returns max(at_end, at_begin), not the sum.
 *   A single allocation cannot split across the wrap boundary.
 *
 * - Zero-size allocation is a no-op returning the current top.
 *   Two consecutive zero-size allocs return the SAME pointer.
 *
 * - Wrap path wasted space: space between old top and heap_size is not
 *   reclaimed.  Inherent ring-buffer fragmentation cost.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "pto_ring_buffer.h"

// =============================================================================
// Helpers
//
// WHITE-BOX: consume_up_to simulates the scheduler consuming tasks by directly
// writing descriptor.packed_buffer_end and advancing last_alive.  This binds
// to the internal tail-derivation mechanism.  If the allocator's reclaim
// protocol changes (e.g. explicit tail field instead of packed_buffer_end),
// this helper and all wrap/reclaim tests must be updated.
// =============================================================================

static void consume_up_to(
    std::vector<PTO2TaskDescriptor> &descriptors, std::atomic<int32_t> &last_alive, void *heap_base,
    int32_t window_size, int32_t new_last_alive, uint64_t heap_tail_offset
) {
    if (new_last_alive <= 0) {
        last_alive.store(new_last_alive, std::memory_order_release);
        return;
    }
    int32_t last_consumed = new_last_alive - 1;
    PTO2TaskDescriptor &descriptor = descriptors[last_consumed & (window_size - 1)];
    descriptor.task_id = PTO2TaskId::make(0, static_cast<uint32_t>(last_consumed));
    descriptor.packed_buffer_end = static_cast<char *>(heap_base) + heap_tail_offset;
    last_alive.store(new_last_alive, std::memory_order_release);
}

static void
publish_allocation(std::vector<PTO2TaskDescriptor> &descriptors, const PTO2TaskAllocResult &result, uint8_t ring_id) {
    PTO2TaskDescriptor &descriptor = descriptors[result.slot];
    descriptor.task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(result.task_id));
    descriptor.packed_buffer_base = result.packed_base;
    descriptor.packed_buffer_end = result.packed_end;
}

// =============================================================================
// Fixture
// =============================================================================

class TaskAllocatorTest : public ::testing::Test {
protected:
    static constexpr int32_t WINDOW_SIZE = 16;
    static constexpr uint64_t HEAP_SIZE = 4096;

    std::vector<PTO2TaskDescriptor> descriptors;
    std::vector<PTO2TaskSlotState> slot_states;
    alignas(64) uint8_t heap_buf[HEAP_SIZE]{};
    std::atomic<int32_t> current_index{0};
    std::atomic<int32_t> last_alive{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2TaskAllocator allocator{};

    void SetUp() override {
        descriptors.assign(WINDOW_SIZE, PTO2TaskDescriptor{});
        slot_states = std::vector<PTO2TaskSlotState>(WINDOW_SIZE);
        std::memset(heap_buf, 0, sizeof(heap_buf));
        current_index.store(0);
        last_alive.store(0);
        error_code.store(PTO2_ERROR_NONE);
        allocator.init(
            descriptors.data(), WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code,
            slot_states.data()
        );
    }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(TaskAllocatorTest, InitialState) {
    EXPECT_EQ(allocator.window_size(), WINDOW_SIZE);
    EXPECT_EQ(allocator.active_count(), 0);
    EXPECT_EQ(allocator.heap_top(), 0u);
    EXPECT_EQ(allocator.heap_capacity(), HEAP_SIZE);
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE);
}

TEST_F(TaskAllocatorTest, AllocNonZeroSize) {
    auto result = allocator.alloc(100);
    ASSERT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 0);
    EXPECT_EQ(result.slot, 0);
    EXPECT_NE(result.packed_base, nullptr);
    // 100 bytes aligned up to PTO2_ALIGN_SIZE (64) = 128
    uint64_t expected_aligned = PTO2_ALIGN_UP(100u, PTO2_ALIGN_SIZE);
    EXPECT_EQ(expected_aligned, 128u);
    EXPECT_EQ(allocator.heap_top(), expected_aligned);
    EXPECT_EQ(
        static_cast<char *>(result.packed_end) - static_cast<char *>(result.packed_base),
        static_cast<ptrdiff_t>(expected_aligned)
    );
}

TEST_F(TaskAllocatorTest, AllocPublishesPendingSlotBeforeCurrentIndex) {
    slot_states[0].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_relaxed);
    std::atomic<PTO2TaskState> observed_state{PTO2_TASK_CONSUMED};
    std::thread observer([&]() {
        while (current_index.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
        observed_state.store(slot_states[0].task_state.load(std::memory_order_acquire), std::memory_order_release);
    });

    auto result = allocator.alloc(100);
    observer.join();

    ASSERT_FALSE(result.failed());
    EXPECT_EQ(result.task_id, 0);
    EXPECT_EQ(current_index.load(std::memory_order_acquire), 1);
    EXPECT_EQ(observed_state.load(std::memory_order_acquire), PTO2_TASK_PENDING);
}

TEST_F(TaskAllocatorTest, SequentialTaskIds) {
    int32_t prev_id = -1;
    for (int i = 0; i < 5; i++) {
        auto result = allocator.alloc(0);
        ASSERT_FALSE(result.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(result.task_id, prev_id + 1) << "Task IDs must be monotonically increasing";
        EXPECT_EQ(result.slot, result.task_id & (WINDOW_SIZE - 1));
        prev_id = result.task_id;
    }
    EXPECT_EQ(allocator.active_count(), 5);
}

TEST_F(TaskAllocatorTest, OutputSizeAlignment) {
    // 1 byte -> aligned to 64
    auto r1 = allocator.alloc(1);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), 64u);

    // Another 33 bytes -> aligned to 64, total 128
    auto r2 = allocator.alloc(33);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator.heap_top(), 128u);

    // Exactly 64 bytes -> stays 64, total 192
    auto r3 = allocator.alloc(64);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(allocator.heap_top(), 192u);
}

TEST_F(TaskAllocatorTest, SlotMappingPowerOfTwoWindow) {
    std::set<int32_t> slots;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, i, 0);
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed());
        EXPECT_EQ(r.slot, r.task_id & (WINDOW_SIZE - 1));
        slots.insert(r.slot);
    }
    EXPECT_EQ(slots.size(), static_cast<size_t>(WINDOW_SIZE))
        << "Every slot should be visited exactly once over one window cycle";
}

TEST_F(TaskAllocatorTest, UpdateHeapTailFromConsumedTask) {
    auto r1 = allocator.alloc(256);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), 256u);

    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE - 256u);

    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 256);

    // Force the allocator to observe the new last_alive by doing another alloc
    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r2.failed());

    // top=256, tail=256: at_end = 4096-256=3840, at_begin = 256
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE - 256u);
}

TEST_F(TaskAllocatorTest, UpdateHeapTailAtTask0) {
    auto r1 = allocator.alloc(64);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.task_id, 0);

    descriptors[0].packed_buffer_end = static_cast<char *>(static_cast<void *>(heap_buf)) + 64;
    last_alive.store(1, std::memory_order_release);

    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.task_id, 1);
}

TEST_F(TaskAllocatorTest, UpdateHeapTailIdempotent) {
    auto r1 = allocator.alloc(128);
    ASSERT_FALSE(r1.failed());

    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 128);

    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r2.failed());
    uint64_t avail_after_first = allocator.heap_available();

    auto r3 = allocator.alloc(0);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(allocator.heap_available(), avail_after_first);
}

TEST_F(TaskAllocatorTest, HeapAvailableTopGeTail) {
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE);

    auto r1 = allocator.alloc(256);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE - 256u);
}

TEST_F(TaskAllocatorTest, HeapAvailableTopLtTail) {
    auto r1 = allocator.alloc(HEAP_SIZE - 64);
    ASSERT_FALSE(r1.failed());
    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, HEAP_SIZE - 64);

    auto r2 = allocator.alloc(128);
    ASSERT_FALSE(r2.failed());
    // top=128, tail=HEAP_SIZE-64: available = (HEAP_SIZE-64) - 128
    EXPECT_EQ(allocator.heap_available(), HEAP_SIZE - 64 - 128);
}

// =============================================================================
// Boundary conditions
// =============================================================================

TEST_F(TaskAllocatorTest, HeapExactFitAtEnd) {
    // Allocate 4032 bytes to leave exactly 64 at end.
    auto r1 = allocator.alloc(HEAP_SIZE - 64);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE - 64u);

    auto r2 = allocator.alloc(64);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);
    EXPECT_EQ(static_cast<char *>(r2.packed_base), reinterpret_cast<char *>(heap_buf) + HEAP_SIZE - 64);
}

// Wrap guard `tail > alloc_size` uses strict > to prevent full/empty ambiguity.
// If the allocation were allowed, heap_top would advance to alloc_size == tail,
// making top == tail.  Because top == tail is the canonical "empty" state, the
// ring could not distinguish "completely full" from "completely empty".
TEST_F(TaskAllocatorTest, HeapWrapGuardRejectsTailEqualsAllocSize) {
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 64);

    auto r2 = allocator.alloc(64);
    EXPECT_TRUE(r2.failed()) << "wrap guard must reject when tail == alloc_size (full/empty ambiguity)";
}

TEST_F(TaskAllocatorTest, HeapWrapAroundSuccess) {
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());

    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 128);

    auto r2 = allocator.alloc(64);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_top(), 64u);
}

// A ring that fully drains can leave top == tail parked at a non-zero offset,
// splitting the free span into [top, size) and [0, tail).  With an allocation
// larger than size/2 neither arc fits it, so without the empty-ring rebase the
// next same-size allocation wedges even though the whole ring is free.
TEST_F(TaskAllocatorTest, HeapEmptyRingRebasesForContiguousAlloc) {
    constexpr uint64_t S = 3000;  // > HEAP_SIZE/2: the ring holds one at a time
    auto r1 = allocator.alloc(S);
    ASSERT_FALSE(r1.failed());
    const uint64_t top_after = allocator.heap_top();

    // Reclaim task 0 fully: tail advances to its extent end == top, draining the
    // ring to empty (top == tail) while parked at a non-zero offset.
    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, top_after);

    // The whole ring is free, so the next same-size allocation must succeed by
    // rebasing to the heap base.
    auto r2 = allocator.alloc(S);
    ASSERT_FALSE(r2.failed()) << "empty ring must place an allocation that fits its whole span";
    EXPECT_EQ(r2.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_top(), top_after);
}

// Linear-gap guard `tail - top > alloc_size` uses strict > for the same reason.
TEST_F(TaskAllocatorTest, HeapLinearGapGuardRejectsExactFit) {
    // Fill most of heap, leaving just 64 at end so next alloc wraps.
    auto r1 = allocator.alloc(HEAP_SIZE - 64);
    ASSERT_FALSE(r1.failed());
    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, HEAP_SIZE - 64);

    // Allocate 128 bytes: space_at_end = 64, not enough -> wrap.
    // tail = HEAP_SIZE-64, which is > 128 -> wraps to beginning.
    auto r2 = allocator.alloc(128);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(allocator.heap_top(), 128u);

    // Now top=128, tail=HEAP_SIZE-64 (top < tail)
    // gap = (HEAP_SIZE-64) - 128 = HEAP_SIZE-192
    // Allocate exactly gap bytes: gap > alloc_size -> FALSE
    uint64_t gap = (HEAP_SIZE - 64) - 128;
    auto r3 = allocator.alloc(gap);
    EXPECT_TRUE(r3.failed()) << "linear-gap guard must reject exact fit (full/empty ambiguity)";
}

TEST_F(TaskAllocatorTest, HeapTopLessThanTailInsufficientSpace) {
    auto r1 = allocator.alloc(HEAP_SIZE - 64);
    ASSERT_FALSE(r1.failed());
    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, HEAP_SIZE - 64);

    auto r2 = allocator.alloc(128);
    ASSERT_FALSE(r2.failed());

    // gap = (HEAP_SIZE-64) - 128. Try to allocate more than gap.
    auto r3 = allocator.alloc(HEAP_SIZE);
    EXPECT_TRUE(r3.failed());
    EXPECT_NE(error_code.load(), 0);
}

// heap_available reports max(at_end, at_begin), not the sum -- a single
// allocation cannot split across the wrap boundary.
TEST_F(TaskAllocatorTest, AvailableReportsMaxNotSum) {
    auto r1 = allocator.alloc(3008);
    ASSERT_FALSE(r1.failed());
    uint64_t actual_top = allocator.heap_top();

    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 1024);

    auto r_probe = allocator.alloc(0);
    ASSERT_FALSE(r_probe.failed());

    uint64_t avail = allocator.heap_available();
    uint64_t at_end = HEAP_SIZE - actual_top;
    uint64_t at_begin = 1024;
    EXPECT_EQ(avail, std::max(at_end, at_begin));
    EXPECT_LT(avail, at_end + at_begin);
}

// Zero-size allocs return the same address and don't advance the top.
TEST_F(TaskAllocatorTest, ZeroSizeAllocationAliased) {
    auto r1 = allocator.alloc(0);
    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r1.failed());
    ASSERT_FALSE(r2.failed());

    EXPECT_EQ(r1.packed_base, r2.packed_base) << "Zero-size allocs return same address";
    EXPECT_EQ(r1.packed_base, r1.packed_end) << "packed_end == packed_base for zero-size";
    EXPECT_EQ(allocator.heap_top(), 0u) << "top doesn't advance for zero-size allocs";
}

TEST_F(TaskAllocatorTest, ReplaysTask2639ZeroExtentHeadAndConsumedTask2640) {
    constexpr int32_t INITIAL_TASK_ID = 2639;
    constexpr uint8_t RING_ID = 3;
    current_index.store(INITIAL_TASK_ID);
    last_alive.store(INITIAL_TASK_ID);
    allocator.init(
        descriptors.data(), WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code,
        slot_states.data(), INITIAL_TASK_ID, RING_ID
    );

    auto zero_head = allocator.alloc(0);
    ASSERT_FALSE(zero_head.failed());
    ASSERT_EQ(zero_head.task_id, 2639);
    publish_allocation(descriptors, zero_head, RING_ID);
    slot_states[zero_head.slot].fanout_count = PTO2_FANOUT_SCOPE_BIT | 64;
    slot_states[zero_head.slot].fanout_refcount.store(PTO2_FANOUT_SCOPE_BIT | 51, std::memory_order_relaxed);
    slot_states[zero_head.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    auto consumed_bytes = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(consumed_bytes.failed());
    ASSERT_EQ(consumed_bytes.task_id, 2640);
    publish_allocation(descriptors, consumed_bytes, RING_ID);
    slot_states[consumed_bytes.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);

    auto next = allocator.alloc(64);

    ASSERT_FALSE(next.failed());
    EXPECT_EQ(next.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(last_alive.load(std::memory_order_acquire), INITIAL_TASK_ID);
    EXPECT_EQ(allocator.heap_reclaim_task_id(), 2641);
    EXPECT_EQ(slot_states[zero_head.slot].task_state.load(std::memory_order_acquire), PTO2_TASK_COMPLETED);
}

TEST_F(TaskAllocatorTest, StopsAtFirstLivePositiveExtent) {
    auto zero_head = allocator.alloc(0);
    ASSERT_FALSE(zero_head.failed());
    publish_allocation(descriptors, zero_head, 0);
    slot_states[zero_head.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    auto live_bytes = allocator.alloc(HEAP_SIZE / 2);
    ASSERT_FALSE(live_bytes.failed());
    publish_allocation(descriptors, live_bytes, 0);
    slot_states[live_bytes.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    auto later_consumed_bytes = allocator.alloc(HEAP_SIZE / 2);
    ASSERT_FALSE(later_consumed_bytes.failed());
    publish_allocation(descriptors, later_consumed_bytes, 0);
    slot_states[later_consumed_bytes.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);

    auto blocked = allocator.alloc(64);

    EXPECT_TRUE(blocked.failed());
    EXPECT_EQ(allocator.heap_reclaim_task_id(), live_bytes.task_id);
    EXPECT_EQ(allocator.heap_tail(), 0u);

    error_code.store(PTO2_ERROR_NONE);
    slot_states[live_bytes.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);
    auto unblocked = allocator.alloc(64);

    ASSERT_FALSE(unblocked.failed());
    EXPECT_EQ(unblocked.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_reclaim_task_id(), later_consumed_bytes.task_id + 1);
    EXPECT_EQ(last_alive.load(std::memory_order_acquire), 0);
}

TEST_F(TaskAllocatorTest, ConcurrentBackpressureReclaimsWithoutLastAliveAdvance) {
    auto zero_head = allocator.alloc(0);
    ASSERT_FALSE(zero_head.failed());
    publish_allocation(descriptors, zero_head, 0);
    slot_states[zero_head.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    auto positive_extent = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(positive_extent.failed());
    publish_allocation(descriptors, positive_extent, 0);
    slot_states[positive_extent.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    PTO2SharedMemoryHeader header{};
    header.orch_error_code.store(PTO2_ERROR_NONE);
    header.sched_error_code.store(PTO2_ERROR_NONE);
    header.orchestrator_reclaim_waiting.store(0);
    std::thread reclaimer([&]() {
        while (header.orchestrator_reclaim_waiting.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
        slot_states[positive_extent.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);
    });

    auto next = allocator.alloc(64, &header, /*scheduler_runs_concurrently=*/true);
    reclaimer.join();

    ASSERT_FALSE(next.failed());
    EXPECT_EQ(next.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(last_alive.load(std::memory_order_acquire), 0);
    EXPECT_EQ(allocator.heap_reclaim_task_id(), positive_extent.task_id + 1);
    EXPECT_EQ(header.orchestrator_reclaim_waiting.load(std::memory_order_acquire), 0);
}

TEST_F(TaskAllocatorTest, DescriptorGenerationMismatchFailsClosed) {
    auto zero_head = allocator.alloc(0);
    ASSERT_FALSE(zero_head.failed());
    publish_allocation(descriptors, zero_head, 1);
    slot_states[zero_head.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    auto consumed_bytes = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(consumed_bytes.failed());
    publish_allocation(descriptors, consumed_bytes, 0);
    slot_states[consumed_bytes.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);

    auto blocked = allocator.alloc(64);

    EXPECT_TRUE(blocked.failed());
    EXPECT_EQ(allocator.heap_reclaim_task_id(), zero_head.task_id);
    EXPECT_EQ(allocator.heap_tail(), 0u);
}

TEST_F(TaskAllocatorTest, DecoupledReclaimAdvancesTailAcrossHeapWrap) {
    auto before_wrap = allocator.alloc(HEAP_SIZE - 64);
    ASSERT_FALSE(before_wrap.failed());
    publish_allocation(descriptors, before_wrap, 0);
    slot_states[before_wrap.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);

    auto zero_head = allocator.alloc(0);
    ASSERT_FALSE(zero_head.failed());
    publish_allocation(descriptors, zero_head, 0);
    slot_states[zero_head.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    EXPECT_EQ(allocator.heap_tail(), HEAP_SIZE - 64);

    auto after_wrap = allocator.alloc(128);
    ASSERT_FALSE(after_wrap.failed());
    publish_allocation(descriptors, after_wrap, 0);
    slot_states[after_wrap.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);
    ASSERT_EQ(after_wrap.packed_base, static_cast<void *>(heap_buf));

    auto next = allocator.alloc(64);

    ASSERT_FALSE(next.failed());
    EXPECT_EQ(allocator.heap_tail(), 128u);
    EXPECT_EQ(allocator.heap_reclaim_task_id(), after_wrap.task_id + 1);
    EXPECT_EQ(next.packed_base, static_cast<void *>(heap_buf + 128));
}

TEST_F(TaskAllocatorTest, HeapCursorDoesNotPermitLiveTaskSlotReuse) {
    constexpr int32_t SMALL_WINDOW_SIZE = 4;
    allocator.init(
        descriptors.data(), SMALL_WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code,
        slot_states.data()
    );

    auto zero_head = allocator.alloc(0);
    ASSERT_FALSE(zero_head.failed());
    publish_allocation(descriptors, zero_head, 0);
    slot_states[zero_head.slot].task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    auto consumed_bytes = allocator.alloc(64);
    ASSERT_FALSE(consumed_bytes.failed());
    publish_allocation(descriptors, consumed_bytes, 0);
    slot_states[consumed_bytes.slot].task_state.store(PTO2_TASK_CONSUMED, std::memory_order_release);

    auto third = allocator.alloc(0);
    ASSERT_FALSE(third.failed());
    publish_allocation(descriptors, third, 0);

    auto overflow = allocator.alloc(0);

    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(allocator.heap_reclaim_task_id(), third.task_id + 1);
    EXPECT_EQ(last_alive.load(std::memory_order_acquire), 0);
    EXPECT_EQ(descriptors[zero_head.slot].task_id, PTO2TaskId::make(0, 0));
    EXPECT_EQ(slot_states[zero_head.slot].task_state.load(std::memory_order_acquire), PTO2_TASK_COMPLETED);
}

// Wrap path: wasted space between old top and heap_size is not reclaimed.
TEST_F(TaskAllocatorTest, WrapPathWastedSpace) {
    auto r1 = allocator.alloc(4000);
    ASSERT_FALSE(r1.failed());
    uint64_t top_after = allocator.heap_top();
    EXPECT_GE(top_after, 4000u);
    EXPECT_LT(top_after, HEAP_SIZE);

    consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, top_after);

    auto r2 = allocator.alloc(128);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.packed_base, static_cast<void *>(heap_buf)) << "Allocation wrapped to beginning";

    uint64_t avail = allocator.heap_available();
    EXPECT_LT(avail, HEAP_SIZE) << "Wasted space at end reduces available capacity";
}

TEST_F(TaskAllocatorTest, AllocExactlyHeapSize) {
    auto r1 = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(allocator.heap_top(), HEAP_SIZE);

    auto r2 = allocator.alloc(64);
    EXPECT_TRUE(r2.failed()) << "No space after full allocation";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(TaskAllocatorTest, AllocLargerThanHeap) {
    auto r = allocator.alloc(HEAP_SIZE * 2);
    EXPECT_TRUE(r.failed()) << "Cannot allocate more than heap size";
    EXPECT_EQ(error_code.load(), PTO2_ERROR_HEAP_RING_DEADLOCK);
}

TEST_F(TaskAllocatorTest, ConcurrentBackpressureWaitsForSchedulerReclaim) {
    auto first = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(first.failed());

    PTO2SharedMemoryHeader header{};
    header.orch_error_code.store(PTO2_ERROR_NONE);
    header.sched_error_code.store(PTO2_ERROR_NONE);
    header.orchestrator_reclaim_waiting.store(0);
    std::atomic<bool> observed_wait{false};

    std::thread reclaimer([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (header.orchestrator_reclaim_waiting.load(std::memory_order_acquire) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        if (header.orchestrator_reclaim_waiting.load(std::memory_order_acquire) != 0) {
            observed_wait.store(true, std::memory_order_release);
            consume_up_to(descriptors, last_alive, heap_buf, WINDOW_SIZE, 1, 128);
        } else {
            header.sched_error_code.store(PTO2_ERROR_SCHEDULER_TIMEOUT, std::memory_order_release);
        }
    });

    auto second = allocator.alloc(64, &header, /*scheduler_runs_concurrently=*/true);
    reclaimer.join();

    EXPECT_TRUE(observed_wait.load(std::memory_order_acquire));
    EXPECT_FALSE(second.failed());
    EXPECT_EQ(second.packed_base, static_cast<void *>(heap_buf));
    EXPECT_EQ(header.orchestrator_reclaim_waiting.load(), 0);
    EXPECT_EQ(error_code.load(), PTO2_ERROR_NONE);
}

TEST_F(TaskAllocatorTest, ConcurrentScopeGatedHeadDefersToSchedulerWatchdog) {
    std::vector<PTO2TaskSlotState> slot_states(WINDOW_SIZE);
    allocator.init(
        descriptors.data(), WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code,
        slot_states.data()
    );

    auto first = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(first.failed());

    PTO2SharedMemoryHeader header{};
    header.orch_error_code.store(PTO2_ERROR_NONE);
    header.sched_error_code.store(PTO2_ERROR_NONE);
    header.orchestrator_reclaim_waiting.store(0);

    PTO2TaskSlotState &head = slot_states[0];
    head.fanout_lock.store(0, std::memory_order_relaxed);
    head.fanout_count = PTO2_FANOUT_SCOPE_BIT;
    head.fanout_refcount.store(0, std::memory_order_relaxed);
    head.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    std::atomic<bool> observed_wait{false};
    std::thread scheduler_failure([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (header.orchestrator_reclaim_waiting.load(std::memory_order_acquire) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        observed_wait.store(
            header.orchestrator_reclaim_waiting.load(std::memory_order_acquire) != 0, std::memory_order_release
        );
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        header.sched_error_code.store(PTO2_ERROR_SCHEDULER_TIMEOUT, std::memory_order_release);
    });

    auto second = allocator.alloc(64, &header, /*scheduler_runs_concurrently=*/true);
    scheduler_failure.join();

    EXPECT_TRUE(observed_wait);
    EXPECT_TRUE(second.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_NONE);
    EXPECT_EQ(header.sched_error_code.load(), PTO2_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(header.orchestrator_reclaim_waiting.load(), 0);
}

TEST_F(TaskAllocatorTest, SchedulerErrorUnwindsConcurrentBackpressureWithoutOrchError) {
    auto first = allocator.alloc(HEAP_SIZE);
    ASSERT_FALSE(first.failed());

    PTO2SharedMemoryHeader header{};
    header.orch_error_code.store(PTO2_ERROR_NONE);
    header.sched_error_code.store(PTO2_ERROR_NONE);
    header.orchestrator_reclaim_waiting.store(0);
    std::atomic<bool> observed_wait{false};

    std::thread scheduler_failure([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (header.orchestrator_reclaim_waiting.load(std::memory_order_acquire) == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        observed_wait.store(
            header.orchestrator_reclaim_waiting.load(std::memory_order_acquire) != 0, std::memory_order_release
        );
        header.sched_error_code.store(PTO2_ERROR_SCHEDULER_TIMEOUT, std::memory_order_release);
    });

    auto second = allocator.alloc(64, &header, /*scheduler_runs_concurrently=*/true);
    scheduler_failure.join();

    EXPECT_TRUE(observed_wait.load(std::memory_order_acquire));
    EXPECT_TRUE(second.failed());
    EXPECT_EQ(header.orchestrator_reclaim_waiting.load(), 0);
    EXPECT_EQ(error_code.load(), PTO2_ERROR_NONE);
}

TEST_F(TaskAllocatorTest, TaskWindowSaturates) {
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        auto r = allocator.alloc(0);
        ASSERT_FALSE(r.failed()) << "Alloc failed at i=" << i;
        EXPECT_EQ(r.task_id, i);
    }
    EXPECT_EQ(allocator.active_count(), WINDOW_SIZE - 1);

    auto overflow = allocator.alloc(0);
    EXPECT_TRUE(overflow.failed());
    EXPECT_EQ(error_code.load(), PTO2_ERROR_FLOW_CONTROL_DEADLOCK);
}

// Task IDs grow monotonically as int32_t. Near INT32_MAX, the same
// signed-overflow concern applies but is cosmetic since we use
// task_id & window_mask for indexing.
TEST_F(TaskAllocatorTest, TaskIdNearInt32Max) {
    current_index.store(INT32_MAX - 2);
    last_alive.store(INT32_MAX - 2);
    allocator.init(
        descriptors.data(), WINDOW_SIZE, &current_index, &last_alive, heap_buf, HEAP_SIZE, &error_code,
        /*slot_states=*/nullptr, /*initial_local_task_id=*/INT32_MAX - 2
    );

    auto r1 = allocator.alloc(0);
    ASSERT_FALSE(r1.failed());
    EXPECT_EQ(r1.task_id, INT32_MAX - 2);
    EXPECT_EQ(r1.slot, (INT32_MAX - 2) & (WINDOW_SIZE - 1));

    auto r2 = allocator.alloc(0);
    ASSERT_FALSE(r2.failed());
    EXPECT_EQ(r2.task_id, INT32_MAX - 1);

    auto r3 = allocator.alloc(0);
    ASSERT_FALSE(r3.failed());
    EXPECT_EQ(r3.task_id, INT32_MAX);
    EXPECT_GE(r3.slot, 0);
    EXPECT_LT(r3.slot, WINDOW_SIZE);
}
