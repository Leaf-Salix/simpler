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
#include <dlfcn.h>
#include <future>
#include <condition_variable>
#include <functional>
#include <thread>
#include "kernel_host_owner.h"

using simpler::kernel::KernelHostOwner;
namespace {
std::string runtime_path;
void *stream = reinterpret_cast<void *>(uintptr_t{1});
class KernelOwner : public ::testing::Test {
protected:
    void *library{};
    void (*reset)(){};
    void (*errors)(int, int, int){};
    int (*launches)(){};
    int (*destroys)(){};
    int (*finalizes)(){};
    uint64_t (*scalar)(){};
    void SetUp() override {
        library = dlopen(runtime_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        ASSERT_NE(library, nullptr);
        reset = reinterpret_cast<decltype(reset)>(dlsym(library, "kernel_test_reset"));
        errors = reinterpret_cast<decltype(errors)>(dlsym(library, "kernel_test_errors"));
        launches = reinterpret_cast<decltype(launches)>(dlsym(library, "kernel_test_launches"));
        destroys = reinterpret_cast<decltype(destroys)>(dlsym(library, "kernel_test_destroys"));
        finalizes = reinterpret_cast<decltype(finalizes)>(dlsym(library, "kernel_test_finalizes"));
        scalar = reinterpret_cast<decltype(scalar)>(dlsym(library, "kernel_test_scalar"));
        ASSERT_NE(reset, nullptr);
        reset();
    }
    void TearDown() override {
        if (library) dlclose(library);
    }
    std::shared_ptr<KernelHostOwner> create() {
        auto owner = std::make_shared<KernelHostOwner>(runtime_path);
        EXPECT_EQ(owner->initialize(0, {1}, {2}, {}, CallConfig{}, 1), 0);
        return owner;
    }
};
TEST_F(KernelOwner, PendingCloseDoesNotCancelAndSnapshotIsOwned) {
    auto owner = create();
    uint64_t value = 41;
    auto call = owner->accept(0, nullptr, 0, &value, 1);
    value = 99;
    EXPECT_EQ(owner->pending(), 1u);
    EXPECT_EQ(owner->close(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(finalizes(), 0);
    EXPECT_EQ(call->invoke(stream), 0);
    EXPECT_EQ(scalar(), 41u);
    EXPECT_EQ(owner->pending(), 0u);
    EXPECT_EQ(owner->close(), 0);
    EXPECT_EQ(call->invoke(stream), 0);
    EXPECT_EQ(launches(), 1);
    EXPECT_EQ(owner->close(), 0);
    EXPECT_EQ(destroys(), 1);
}
TEST_F(KernelOwner, CallbackCopiesAndThrowAfterCopyKeepOneTicket) {
    auto owner = create();
    std::function<int()> queued;
    {
        auto call = owner->accept(0, nullptr, 0, nullptr, 0);
        auto callback = [call] {
            return call->invoke(stream);
        };
        try {
            queued = callback;
            throw std::runtime_error("submission failed after copy");
        } catch (const std::runtime_error &) {}
    }
    EXPECT_EQ(owner->pending(), 1u);
    EXPECT_EQ(queued(), 0);
    EXPECT_EQ(queued(), 0);
    EXPECT_EQ(launches(), 1);
    EXPECT_EQ(owner->pending(), 0u);
    EXPECT_EQ(owner->close(), 0);
}
TEST_F(KernelOwner, DirectLaunchCannotOvertakeQueuedInvocation) {
    auto owner = create();
    auto queued = owner->accept(0, nullptr, 0, nullptr, 0);
    ChipStorageTaskArgs args{};
    EXPECT_EQ(owner->launch(0, &args, stream), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(launches(), 0);
    EXPECT_EQ(queued->invoke(stream), 0);
    EXPECT_EQ(owner->launch(0, &args, stream), 0);
    EXPECT_EQ(launches(), 2);
    EXPECT_EQ(owner->close(), 0);
    EXPECT_FALSE(owner->ready());
    EXPECT_EQ(owner->launch(0, &args, stream), PTO_RUNTIME_ERR_INVALID_STATE);
}
TEST_F(KernelOwner, LastDiscardSettlesPending) {
    auto owner = create();
    auto call = owner->accept(0, nullptr, 0, nullptr, 0);
    auto copy = call;
    call.reset();
    EXPECT_EQ(owner->pending(), 1u);
    copy.reset();
    EXPECT_EQ(owner->pending(), 0u);
    EXPECT_EQ(owner->close(), 0);
}
TEST_F(KernelOwner, InitFailureAndCloseFailureRemainRetryable) {
    auto owner = std::make_shared<KernelHostOwner>(runtime_path);
    errors(-1000, -1001, 0);
    EXPECT_EQ(owner->initialize(0, {1}, {2}, {}, CallConfig{}, 1), -1000);
    EXPECT_EQ(owner->close(), -1001);
    EXPECT_EQ(destroys(), 0);
    EXPECT_THROW(owner->accept(0, nullptr, 0, nullptr, 0), std::runtime_error);
    errors(0, 0, 0);
    EXPECT_EQ(owner->initialize(0, {1}, {2}, {}, CallConfig{}, 2), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(owner->close(), 0);
    EXPECT_EQ(destroys(), 1);
}
TEST_F(KernelOwner, NativeFailureAndRecordFailureAreNotRetried) {
    auto owner = create();
    errors(0, 0, -1001);
    auto call = owner->accept(0, nullptr, 0, nullptr, 0);
    EXPECT_EQ(call->invoke(stream), -1001);
    EXPECT_EQ(call->invoke(stream), -1001);
    EXPECT_EQ(launches(), 1);
    auto blocked = owner->accept(0, nullptr, 0, nullptr, 0);
    auto before = [](void *) noexcept {
        return -77;
    };
    EXPECT_EQ(blocked->invoke(stream, before), -77);
    EXPECT_EQ(blocked->invoke(stream), -77);
    EXPECT_EQ(launches(), 1);
    EXPECT_EQ(owner->close(), 0);
}
TEST_F(KernelOwner, SameTicketConcurrentInvokeRunsOnce) {
    auto owner = create();
    auto call = owner->accept(0, nullptr, 0, nullptr, 0);
    auto a = std::async(std::launch::async, [&] {
        return call->invoke(stream);
    });
    auto b = std::async(std::launch::async, [&] {
        return call->invoke(stream);
    });
    EXPECT_EQ(a.get(), 0);
    EXPECT_EQ(b.get(), 0);
    EXPECT_EQ(launches(), 1);
    EXPECT_EQ(owner->close(), 0);
}
TEST_F(KernelOwner, NativeExceptionSettlesTicketWithoutRetry) {
    auto owner = create();
    errors(0, 0, -12345);
    auto call = owner->accept(0, nullptr, 0, nullptr, 0);
    EXPECT_EQ(call->invoke(stream), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(call->invoke(stream), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(launches(), 1);
    EXPECT_EQ(owner->pending(), 0u);
    EXPECT_EQ(owner->close(), 0);
}
TEST_F(KernelOwner, CloseSerializesWithActiveNativeSubmission) {
    auto owner = create();
    auto call = owner->accept(0, nullptr, 0, nullptr, 0);
    struct Barrier {
        std::promise<void> entered;
        std::shared_future<void> proceed;
    };
    std::promise<void> proceed;
    Barrier barrier{{}, proceed.get_future().share()};
    auto entered = barrier.entered.get_future();
    auto before = [](void *data) noexcept {
        auto &barrier = *static_cast<Barrier *>(data);
        barrier.entered.set_value();
        barrier.proceed.wait();
        return 0;
    };
    auto invocation = std::async(std::launch::async, [&] {
        return call->invoke(stream, before, &barrier);
    });
    entered.wait();
    EXPECT_EQ(finalizes(), 0);
    std::promise<void> attempting_close;
    auto attempted = attempting_close.get_future();
    auto close = std::async(std::launch::async, [&] {
        attempting_close.set_value();
        return owner->close();
    });
    attempted.wait();
    // The invocation holds the owner gate until native enqueue returns.
    EXPECT_EQ(finalizes(), 0);
    proceed.set_value();
    EXPECT_EQ(invocation.get(), 0);
    EXPECT_EQ(close.get(), 0);
    EXPECT_EQ(launches(), 1);
    EXPECT_EQ(destroys(), 1);
}
TEST_F(KernelOwner, CallbackKeepsOwnerAfterPythonHandleDrop) {
    auto owner = create();
    std::weak_ptr<KernelHostOwner> weak = owner;
    auto call = owner->accept(0, nullptr, 0, nullptr, 0);
    owner.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(call->invoke(stream), 0);
    auto retained = weak.lock();
    ASSERT_NE(retained, nullptr);
    EXPECT_EQ(retained->close(), 0);
}
TEST_F(KernelOwner, InvalidInputDoesNotAdmit) {
    auto owner = create();
    EXPECT_THROW(owner->accept(-1, nullptr, 0, nullptr, 0), std::invalid_argument);
    EXPECT_THROW(owner->accept(0, nullptr, 1, nullptr, 0), std::invalid_argument);
    EXPECT_EQ(owner->pending(), 0u);
    EXPECT_EQ(owner->close(), 0);
}
}  // namespace
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (argc != 2) return 2;
    runtime_path = argv[1];
    return RUN_ALL_TESTS();
}
