#include <gtest/gtest.h>

#include <thread>

#include "host/kernel_submission_state.h"

TEST(KernelSubmissionState, TracksCallerTailOnlyAfterCommit) {
    KernelSubmissionState state;
    {
        auto lease = state.lock();
        EXPECT_FALSE(state.has_tail());
        EXPECT_EQ(state.previous_caller(), 0u);
        state.commit_tail(0x1234u);
        EXPECT_TRUE(state.has_tail());
        EXPECT_EQ(state.previous_caller(), 0x1234u);
        EXPECT_FALSE(state.closing());
    }
    EXPECT_FALSE(state.closing());
}

TEST(KernelSubmissionState, CloseIsStickyWithinSubmissionLease) {
    KernelSubmissionState state;
    auto lease = state.lock();
    state.begin_close();
    EXPECT_TRUE(state.closing());
    state.commit_tail(0x5678u);
    EXPECT_EQ(state.previous_caller(), 0x5678u);
    EXPECT_TRUE(state.has_tail());
}

TEST(KernelSubmissionState, TailCanBeClearedAfterNativeQuiescence) {
    KernelSubmissionState state;
    state.commit_tail(0x9abcu);
    ASSERT_TRUE(state.has_tail());
    state.clear_tail();
    EXPECT_FALSE(state.has_tail());
    EXPECT_EQ(state.previous_caller(), 0u);
}

TEST(KernelSubmissionState, MutexSerializesLeases) {
    KernelSubmissionState state;
    auto first = state.lock();
    bool second_acquired = false;
    std::thread waiter([&] {
        auto second = state.lock();
        second_acquired = true;
    });
    std::this_thread::yield();
    EXPECT_FALSE(second_acquired);
    first.unlock();
    waiter.join();
    EXPECT_TRUE(second_acquired);
}
