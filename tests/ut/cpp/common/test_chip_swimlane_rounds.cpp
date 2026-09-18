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

#include "aicpu/chip_swimlane_collector_aicpu.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

extern "C" void set_platform_chip_swimlane_base(uint64_t);
extern "C" void set_chip_swimlane_enabled(bool);

namespace {

struct Region {
    ChipSwimlaneDataHeader header{};
    ChipSwimlaneAicpuTaskPool tasks[PLATFORM_MAX_CORES]{};
    ChipSwimlaneAicoreTaskPool cores[PLATFORM_MAX_CORES]{};
    ChipSwimlaneAicpuSchedPhasePool sched[PLATFORM_MAX_AICPU_THREADS]{};
    ChipSwimlaneAicpuOrchPhasePool orch[PLATFORM_MAX_AICPU_THREADS]{};
};

class ChipSwimlaneRoundsTest : public testing::TestWithParam<ChipSwimlaneLevel> {
protected:
    std::unique_ptr<Region> region = std::make_unique<Region>();
    std::vector<ChipSwimlaneAicpuTaskBuffer> tasks{PLATFORM_PROF_SLOT_COUNT};
    std::vector<ChipSwimlaneAicoreTaskBuffer> cores{PLATFORM_PROF_SLOT_COUNT};
    std::vector<ChipSwimlaneAicpuSchedPhaseBuffer> sched{PLATFORM_PROF_SLOT_COUNT};
    std::vector<ChipSwimlaneAicpuOrchPhaseBuffer> orch{PLATFORM_PROF_SLOT_COUNT};

    template <typename Pool, typename Buffer>
    void seed(Pool &pool, std::vector<Buffer> &buffers) {
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            pool.free_queue.buffer_ptrs[i] = reinterpret_cast<uint64_t>(&buffers[i]);
        }
        pool.free_queue.tail = buffers.size();
    }

    void SetUp() override {
        ASSERT_EQ(sizeof(Region), calc_perf_data_size_with_phases());
        region->header.chip_swimlane_level = static_cast<uint32_t>(GetParam());
        seed(region->tasks[0], tasks);
        seed(region->cores[0], cores);
        seed(region->sched[0], sched);
        seed(region->orch[0], orch);
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(region.get()));
        set_chip_swimlane_enabled(true);
    }

    void TearDown() override {
        set_chip_swimlane_enabled(false);
        set_platform_chip_swimlane_base(0);
    }

    void init() {
        chip_swimlane_aicpu_init(1);
        chip_swimlane_aicpu_init_phase(1, 1, 1);
        chip_swimlane_aicpu_set_orch_thread_idx(0);
    }

    void flush() {
        int core = 0;
        chip_swimlane_aicpu_flush(0, &core, 1);
        chip_swimlane_aicpu_flush_sched_phase_buffer(0);
        chip_swimlane_aicpu_flush_orch_phase_buffer(0);
    }

    struct Collected {
        uint32_t task = 0;
        uint32_t core = 0;
        uint32_t sched = 0;
        uint32_t orch = 0;
        std::vector<uint32_t> core_counts;
        std::vector<uint32_t> core_seqs;
    };

    Collected drain() {
        Collected result;
        auto &header = region->header;
        while (header.queue_heads[0] != header.queue_tails[0]) {
            auto entry = header.queues[0][header.queue_heads[0]];
            ChipSwimlaneFreeQueue *free_queue = nullptr;
            switch (entry.kind) {
            case ChipSwimlaneBufferKind::AicpuTask:
                result.task += reinterpret_cast<ChipSwimlaneAicpuTaskBuffer *>(entry.buffer_ptr)->count;
                free_queue = &region->tasks[entry.core_index].free_queue;
                break;
            case ChipSwimlaneBufferKind::AicoreTask: {
                auto count = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(entry.buffer_ptr)->count;
                result.core += count;
                result.core_counts.push_back(count);
                result.core_seqs.push_back(entry.buffer_seq);
                free_queue = &region->cores[entry.core_index].free_queue;
                break;
            }
            case ChipSwimlaneBufferKind::AicpuSchedPhase:
                result.sched += reinterpret_cast<ChipSwimlaneAicpuSchedPhaseBuffer *>(entry.buffer_ptr)->count;
                free_queue = &region->sched[entry.core_index].free_queue;
                break;
            case ChipSwimlaneBufferKind::AicpuOrchPhase:
                result.orch += reinterpret_cast<ChipSwimlaneAicpuOrchPhaseBuffer *>(entry.buffer_ptr)->count;
                free_queue = &region->orch[entry.core_index].free_queue;
                break;
            }
            EXPECT_LT(free_queue->tail - free_queue->head, PLATFORM_PROF_SLOT_COUNT);
            free_queue->buffer_ptrs[free_queue->tail % PLATFORM_PROF_SLOT_COUNT] = entry.buffer_ptr;
            free_queue->tail++;
            header.queue_heads[0] = (header.queue_heads[0] + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
        }
        return result;
    }
};

TEST_P(ChipSwimlaneRoundsTest, IdlePoolsKeepTheirBuffersAcrossLaunches) {
    for (int round = 0; round < 4 * PLATFORM_PROF_SLOT_COUNT; ++round) {
        SCOPED_TRACE(round);
        init();
        ASSERT_EQ(region->tasks[0].head.current_buf_ptr, reinterpret_cast<uint64_t>(&tasks[0]));
        ASSERT_EQ(region->cores[0].head.current_buf_ptr, reinterpret_cast<uint64_t>(&cores[0]));
        ASSERT_EQ(region->sched[0].head.current_buf_ptr, reinterpret_cast<uint64_t>(&sched[0]));
        ASSERT_EQ(region->orch[0].head.current_buf_ptr, reinterpret_cast<uint64_t>(&orch[0]));
        flush();
        EXPECT_EQ(region->header.queue_tails[0], 0u);
    }
    EXPECT_EQ(region->tasks[0].free_queue.head, 1u);
    EXPECT_EQ(region->cores[0].free_queue.head, 1u);
    EXPECT_EQ(region->sched[0].free_queue.head, 1u);
    EXPECT_EQ(region->orch[0].free_queue.head, 1u);
    init();
    chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 1);
    ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 10, 20), 0);
    chip_swimlane_aicpu_record_dummy_task(0, 20, 0, 0);
    chip_swimlane_aicpu_record_orch_phase(1, 2, 0, 0);
    flush();
    auto collected = drain();
    EXPECT_EQ(collected.task, 1u);
    EXPECT_EQ(collected.core, 1u);
    EXPECT_EQ(collected.sched, 1u);
    EXPECT_EQ(collected.orch, 1u);
}

TEST_P(ChipSwimlaneRoundsTest, PartialFlushCountsOnlyTheCurrentLaunch) {
    for (uint32_t round = 0; round < 4 * PLATFORM_PROF_SLOT_COUNT; ++round) {
        SCOPED_TRACE(round);
        init();
        chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 1);
        ASSERT_EQ(chip_swimlane_aicpu_complete_task(0, 0, 1, 10, 20), 0);
        chip_swimlane_aicpu_record_dummy_task(0, 20, 0, 0);
        chip_swimlane_aicpu_record_orch_phase(1, 2, 0, 0);
        flush();
        auto collected = drain();
        EXPECT_EQ(collected.task, 1u);
        EXPECT_EQ(collected.core, 1u);
        EXPECT_EQ(collected.sched, 1u);
        EXPECT_EQ(collected.orch, 1u);
        EXPECT_EQ(collected.core_seqs, std::vector<uint32_t>{round});
        EXPECT_EQ(region->tasks[0].head.total_record_count, round + 1);
        EXPECT_EQ(region->cores[0].head.total_record_count, round + 1);
    }
}

TEST_P(ChipSwimlaneRoundsTest, RotationAfterPartialLaunchFlushHasAnExactTail) {
    init();
    chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 1);
    flush();
    drain();

    init();
    for (uint32_t task = 0; task < PLATFORM_AICORE_BUFFER_SIZE + 1; ++task) {
        chip_swimlane_aicpu_on_aicore_dispatch(0, 0, task + 1);
        chip_swimlane_aicpu_on_aicore_ack(0, 0, task + 1);
    }
    flush();
    auto collected = drain();
    EXPECT_EQ(collected.core_counts, (std::vector<uint32_t>{PLATFORM_AICORE_BUFFER_SIZE, 1}));
    EXPECT_EQ(collected.core_seqs, (std::vector<uint32_t>{1, 2}));
    EXPECT_EQ(region->cores[0].head.total_record_count, PLATFORM_AICORE_BUFFER_SIZE + 2u);
}

TEST_P(ChipSwimlaneRoundsTest, FailedRotationKeepsFullBufferAndRecoveryHasAnExactTail) {
    init();
    auto &queue = region->cores[0].free_queue;
    uint32_t slot = queue.head % PLATFORM_PROF_SLOT_COUNT;
    uint64_t replacement = queue.buffer_ptrs[slot];
    queue.buffer_ptrs[slot] = 0;
    for (uint32_t task = 0; task < 2 * PLATFORM_AICORE_BUFFER_SIZE; ++task) {
        chip_swimlane_aicpu_on_aicore_dispatch(0, 0, task + 1);
    }
    EXPECT_EQ(region->cores[0].head.current_buf_ptr, reinterpret_cast<uint64_t>(&cores[0]));
    queue.buffer_ptrs[slot] = replacement;
    chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 2 * PLATFORM_AICORE_BUFFER_SIZE + 1);
    chip_swimlane_aicpu_on_aicore_ack(0, 0, 2 * PLATFORM_AICORE_BUFFER_SIZE + 1);
    flush();
    auto collected = drain();
    EXPECT_EQ(collected.core_counts, (std::vector<uint32_t>{PLATFORM_AICORE_BUFFER_SIZE, 1}));
    EXPECT_EQ(region->cores[0].head.total_record_count, 2 * PLATFORM_AICORE_BUFFER_SIZE + 1u);
}

TEST_P(ChipSwimlaneRoundsTest, DisabledWindowClearsCachedPhaseEnablement) {
    init();
    chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 1);
    chip_swimlane_aicpu_record_dummy_task(0, 20, 0, 0);
    chip_swimlane_aicpu_record_orch_phase(1, 2, 0, 0);
    flush();
    drain();
    set_chip_swimlane_enabled(false);
    EXPECT_EQ(get_chip_swimlane_level(), ChipSwimlaneLevel::DISABLED);
    chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 1);
    chip_swimlane_aicpu_record_dummy_task(0, 20, 0, 0);
    chip_swimlane_aicpu_record_orch_phase(1, 2, 0, 0);
    flush();
    EXPECT_EQ(region->header.queue_heads[0], region->header.queue_tails[0]);
    EXPECT_EQ(region->cores[0].head.total_record_count, 1u);
    EXPECT_EQ(region->sched[0].head.total_record_count, 1u);
    EXPECT_EQ(region->orch[0].head.total_record_count, 1u);
    set_chip_swimlane_enabled(true);
    init();
    EXPECT_EQ(get_chip_swimlane_level(), GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Levels, ChipSwimlaneRoundsTest, testing::Values(ChipSwimlaneLevel::TASK_TIMING, ChipSwimlaneLevel::ORCH_PHASES)
);

}  // namespace
