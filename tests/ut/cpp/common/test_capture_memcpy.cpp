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

#include <cstring>
#include <utility>

#include "host/capture_memcpy.h"

namespace {
aclmdlRICaptureMode thread_mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
int exchange_calls = 0;
int copy_calls = 0;
int fail_exchange = 0;
int copy_error = 0;

class CaptureMemcpy : public testing::Test {
protected:
    void SetUp() override {
        thread_mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
        exchange_calls = 0;
        copy_calls = 0;
        fail_exchange = 0;
        copy_error = 0;
    }
};
}  // namespace

extern "C" aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode) {
    if (++exchange_calls == fail_exchange) return -71;
    std::swap(thread_mode, *mode);
    return ACL_SUCCESS;
}

extern "C" aclError aclrtMemcpy(void *dst, size_t dst_bytes, const void *src, size_t src_bytes, aclrtMemcpyKind kind) {
    ++copy_calls;
    EXPECT_EQ(thread_mode, ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
    EXPECT_EQ(kind, ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_GE(dst_bytes, src_bytes);
    if (copy_error != 0) return copy_error;
    std::memcpy(dst, src, src_bytes);
    return ACL_SUCCESS;
}

TEST_F(CaptureMemcpy, CopiesImmediatelyAndRestoresEveryMode) {
    for (auto mode :
         {ACL_MODEL_RI_CAPTURE_MODE_GLOBAL, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL,
          ACL_MODEL_RI_CAPTURE_MODE_RELAXED}) {
        thread_mode = mode;
        int src = 42;
        int dst = 0;
        ASSERT_EQ(capture_memcpy_h2d(&dst, sizeof(dst), &src, sizeof(src)), 0);
        EXPECT_EQ(dst, src);
        EXPECT_EQ(thread_mode, mode);
    }
    EXPECT_EQ(copy_calls, 3);
    EXPECT_EQ(exchange_calls, 6);
}

TEST_F(CaptureMemcpy, FailedExchangeDoesNotCopyOrRestore) {
    fail_exchange = 1;
    int value = 0;
    EXPECT_EQ(capture_memcpy_h2d(&value, sizeof(value), &value, sizeof(value)), -71);
    EXPECT_EQ(copy_calls, 0);
    EXPECT_EQ(exchange_calls, 1);
    EXPECT_EQ(thread_mode, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
}

TEST_F(CaptureMemcpy, FailedCopyRestoresModeAndReturnsError) {
    copy_error = -72;
    int value = 0;
    EXPECT_EQ(capture_memcpy_h2d(&value, sizeof(value), &value, sizeof(value)), -72);
    EXPECT_EQ(exchange_calls, 2);
    EXPECT_EQ(thread_mode, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
}

TEST_F(CaptureMemcpy, FailedRestoreCannotReportSuccess) {
    fail_exchange = 2;
    int value = 0;
    EXPECT_EQ(capture_memcpy_h2d(&value, sizeof(value), &value, sizeof(value)), -71);
    EXPECT_EQ(copy_calls, 1);
    EXPECT_EQ(exchange_calls, 2);
}
