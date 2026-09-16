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

#include <array>
#include "host/kernel_context_revoke.h"

namespace {
struct RevokeFixture {
    std::array<int, 5> calls{};
    int fail{-1};
    int call(int index) {
        ++calls[index];
        return index == fail ? -71 : 0;
    }
    KernelRevokeOps ops() {
        return {
            this,
            [](void *p) noexcept {
                return static_cast<RevokeFixture *>(p)->call(0);
            },
            [](void *p) noexcept {
                return static_cast<RevokeFixture *>(p)->call(1);
            },
            [](void *p) noexcept {
                return static_cast<RevokeFixture *>(p)->call(2);
            },
            [](void *p) noexcept {
                return static_cast<RevokeFixture *>(p)->call(3);
            },
            [](void *p) noexcept {
                return static_cast<RevokeFixture *>(p)->call(4);
            }
        };
    }
};

TEST(KernelContextRevokeTest, SuccessfulCloseWaitsAndConfirmsInOneCall) {
    KernelContextRevoke revoke;
    RevokeFixture f;
    revoke.registration_may_exist();
    EXPECT_EQ(revoke.advance(f.ops()), 0);
    EXPECT_TRUE(revoke.confirmed());
    EXPECT_EQ(f.calls, (std::array<int, 5>{1, 1, 1, 1, 1}));
    EXPECT_EQ(revoke.advance({}), 0);
    EXPECT_EQ(f.calls, (std::array<int, 5>{1, 1, 1, 1, 1}));
}

TEST(KernelContextRevokeTest, WaitFailureKeepsAllOwnershipAndDoesNotResubmit) {
    KernelContextRevoke revoke;
    RevokeFixture f;
    EXPECT_EQ(revoke.advance({}), 0);
    revoke.registration_may_exist();
    EXPECT_FALSE(revoke.confirmed());
    f.fail = 3;
    EXPECT_EQ(revoke.advance(f.ops()), -71);
    EXPECT_EQ(revoke.advance(f.ops()), -71);
    EXPECT_EQ(f.calls, (std::array<int, 5>{1, 1, 1, 2, 0}));
    EXPECT_FALSE(revoke.confirmed());
    f.fail = -1;
    EXPECT_EQ(revoke.advance(f.ops()), 0);
    EXPECT_TRUE(revoke.confirmed());
    EXPECT_EQ(revoke.advance({}), 0);
    EXPECT_EQ(f.calls, (std::array<int, 5>{1, 1, 1, 3, 1}));
}

TEST(KernelContextRevokeTest, RetryResumesAtFirstUncommittedStep) {
    for (int failure = 0; failure < 5; ++failure) {
        SCOPED_TRACE(failure);
        KernelContextRevoke revoke;
        RevokeFixture f;
        revoke.registration_may_exist();
        f.fail = failure;
        EXPECT_EQ(revoke.advance(f.ops()), -71);
        EXPECT_FALSE(revoke.confirmed());
        for (int step = 0; step < 5; ++step)
            EXPECT_EQ(f.calls[step], step <= failure ? 1 : 0);
        f.fail = -1;
        EXPECT_EQ(revoke.advance(f.ops()), 0);
        EXPECT_TRUE(revoke.confirmed());
        for (int step = 0; step < 3; ++step)
            EXPECT_EQ(f.calls[step], step == failure ? 2 : 1);
    }
}

TEST(KernelContextRevokeTest, NeverQueriesUnrecordedEventOrAcceptsInvalidReceipt) {
    KernelContextRevoke revoke;
    RevokeFixture f;
    revoke.registration_may_exist();
    EXPECT_EQ(revoke.advance({}), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    f.fail = 2;
    EXPECT_EQ(revoke.advance(f.ops()), -71);
    EXPECT_EQ(f.calls[3], 0);
    f.fail = 4;
    EXPECT_EQ(revoke.advance(f.ops()), -71);
    EXPECT_EQ(revoke.advance(f.ops()), -71);
    EXPECT_FALSE(revoke.confirmed());
    EXPECT_EQ(f.calls[0], 1);
    EXPECT_EQ(f.calls[1], 1);
    EXPECT_EQ(f.calls[2], 2);
}
}  // namespace
