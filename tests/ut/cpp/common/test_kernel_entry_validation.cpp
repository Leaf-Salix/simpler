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

#include <cstdlib>
#include <vector>

#include "host/kernel_entry_validation.h"

namespace {

int dummy_ctx_storage = 0;
void *const kCtx = &dummy_ctx_storage;
const uint8_t kBinary[4] = {1, 2, 3, 4};
const int kConfigStorage = 0;
const void *const kConfig = &kConfigStorage;
int dummy_stream_storage = 0;
void *const kStream = &dummy_stream_storage;
alignas(ChipCallable) const unsigned char kCallableImage[sizeof(ChipCallable)] = {};

TEST(KernelEntryValidation, BinarySpanRequiresPointerAndSizeTogether) {
    EXPECT_TRUE(kernel_binary_span_is_consistent(nullptr, 0));
    EXPECT_TRUE(kernel_binary_span_is_consistent(kBinary, sizeof(kBinary)));
    EXPECT_FALSE(kernel_binary_span_is_consistent(nullptr, 4));
    EXPECT_FALSE(kernel_binary_span_is_consistent(kBinary, 0));
}

TEST(KernelEntryValidation, InitAcceptsConsistentArgs) {
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        0
    );
}

TEST(KernelEntryValidation, InitRejectsEachStructuralViolation) {
    EXPECT_EQ(
        validate_kernel_init_args(
            nullptr, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1
        ),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, nullptr, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, -1, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 0, kConfig, 0),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    // One inconsistent span per position, in both directions.
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, nullptr, 4, kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, 0, kBinary, sizeof(kBinary), nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), nullptr, 4, nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, 0, nullptr, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), kBinary, 0, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_init_args(kCtx, 0, kBinary, sizeof(kBinary), kBinary, sizeof(kBinary), nullptr, 4, kConfig, 1),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelEntryValidation, PrepareCallableChecksOutputPointerAndImageSize) {
    int32_t id = -1;
    EXPECT_EQ(validate_kernel_prepare_callable_args(kCtx, kCallableImage, sizeof(ChipCallable), &id), 0);
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(nullptr, kCallableImage, sizeof(ChipCallable), &id),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, nullptr, sizeof(ChipCallable), &id),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, kCallableImage, sizeof(ChipCallable), nullptr),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, kCallableImage, sizeof(ChipCallable) - 1, &id),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(
        validate_kernel_prepare_callable_args(kCtx, kCallableImage + 1, sizeof(ChipCallable), &id),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelEntryValidation, CallableImageRejectsTruncatedVariableTail) {
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, kBinary, sizeof(kBinary));
    const int32_t child_id = 7;
    const std::vector<uint8_t> children[] = {child};
    const auto image = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch", kBinary, sizeof(kBinary), &child_id, children, 1, ""
    );
    ASSERT_GT(image.size(), sizeof(ChipCallable));
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), 0);
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size() - 1), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

TEST(KernelEntryValidation, CallableImageRejectsOutOfRangeChildMetadata) {
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, kBinary, sizeof(kBinary));
    const int32_t child_id = 7;
    const std::vector<uint8_t> children[] = {child};
    auto image = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch", nullptr, 0, &child_id, children, 1, ""
    );
    auto *callable = reinterpret_cast<ChipCallable *>(image.data());
    callable->child_offsets_[0] = static_cast<uint32_t>(image.size());
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

TEST(KernelEntryValidation, LaunchChecksPointersAndIdRange) {
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 0, kCallableImage, kStream), 0);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 8191, kCallableImage, kStream), 0);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 8192, kCallableImage, kStream), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_launch_args(nullptr, 0, kCallableImage, kStream), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 0, nullptr, kStream), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, 0, kCallableImage, nullptr), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(validate_kernel_launch_args(kCtx, -1, kCallableImage, kStream), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(
        validate_kernel_launch_args(kCtx, MAX_REGISTERED_CALLABLE_IDS, kCallableImage, kStream),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
}

TEST(KernelEntryValidation, CallableImageRejectsInvalidAndDuplicateChildFunctionIds) {
    const auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, kBinary, sizeof(kBinary));
    const int32_t ids[] = {0, KERNEL_MAX_FUNC_ID - 1};
    const std::vector<uint8_t> children[] = {child, child};
    auto image =
        make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(nullptr, 0, "orch", nullptr, 0, ids, children, 2, "");
    auto *callable = reinterpret_cast<ChipCallable *>(image.data());
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), 0);
    for (int32_t id : {-1, KERNEL_MAX_FUNC_ID, MAX_REGISTERED_CALLABLE_IDS - 1}) {
        callable->child_func_ids_[1] = id;
        EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    }
    callable->child_func_ids_[1] = 0;
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    callable->child_func_ids_[0] = KERNEL_MAX_FUNC_ID - 1;
    EXPECT_EQ(validate_kernel_callable_image(image.data(), image.size()), 0);
}

}  // namespace
