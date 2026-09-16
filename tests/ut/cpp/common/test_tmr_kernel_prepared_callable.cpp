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
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "tensormap_and_ringbuffer/kernel_prepared_callable.h"

namespace {
using namespace simpler::tmr;

// Owned Host bytes exercise registration parsing and borrowed table addresses.
// No HBM allocation, executable device image or production provider is created.
std::vector<uint8_t> make_image() {
    const uint8_t binary[] = {1, 2, 3, 4};
    const std::array<ArgDirection, 4> signature{
        ArgDirection::IN, ArgDirection::OUT, ArgDirection::SCALAR, ArgDirection::SCALAR
    };
    const std::array<int32_t, 2> function_ids{2, 7};
    const std::array<std::vector<uint8_t>, 2> children{
        make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary)),
        make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary))
    };
    auto image = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        signature.data(), signature.size(), "orch", binary, sizeof(binary), function_ids.data(), children.data(),
        children.size(), "config"
    );
    auto *chip = reinterpret_cast<ChipCallable *>(image.data());
    for (int32_t i = 0; i < chip->child_count(); ++i) {
        auto *child = reinterpret_cast<CoreCallable *>(chip->storage_ + chip->child_offset(i));
        child->set_resolved_addr(reinterpret_cast<uint64_t>(child->binary_data()));
    }
    return image;
}

void expect_equal(const PreparedKernelCallable &actual, const PreparedKernelCallable &expected) {
    EXPECT_EQ(actual.device_address, expected.device_address);
    EXPECT_EQ(actual.bytes, expected.bytes);
    EXPECT_EQ(actual.identity.callable_id, expected.identity.callable_id);
    EXPECT_EQ(actual.identity.tensor_count, expected.identity.tensor_count);
    EXPECT_EQ(actual.identity.scalar_count, expected.identity.scalar_count);
    EXPECT_EQ(actual.functions, expected.functions);
}

class TmrKernelPreparedCallableTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(reinterpret_cast<uintptr_t>(image.data()) % alignof(ChipCallable), 0u);
        resident = {13, reinterpret_cast<uint64_t>(image.data()), image.size(), 3, 0};
        ASSERT_TRUE(make_prepared_kernel_callable(resident, &prepared));
    }

    ChipCallable &chip() { return *reinterpret_cast<ChipCallable *>(image.data()); }
    CoreCallable &child(int32_t index) {
        return *reinterpret_cast<CoreCallable *>(chip().storage_ + chip().child_offset(index));
    }

    void reject(const TmrCallableRegistrationArgs &candidate) {
        auto out = prepared;
        const auto *storage = out.functions.data();
        const size_t capacity = out.functions.capacity();
        EXPECT_FALSE(make_prepared_kernel_callable(candidate, &out));
        expect_equal(out, prepared);
        EXPECT_EQ(out.functions.data(), storage);
        EXPECT_EQ(out.functions.capacity(), capacity);
    }
    void reject() { reject(resident); }

    std::vector<uint8_t> image{make_image()};
    TmrCallableRegistrationArgs resident{};
    PreparedKernelCallable prepared{};
};

TEST_F(TmrKernelPreparedCallableTest, CountsComeFromSignatureAndIgnoreAbiPadding) {
    EXPECT_EQ(prepared.identity.callable_id, 3);
    EXPECT_EQ(prepared.identity.tensor_count, 2);
    EXPECT_EQ(prepared.identity.scalar_count, 2);
    const auto before = prepared;
    constexpr size_t padding = offsetof(ChipCallable, config_name_len_) + sizeof(uint32_t);
    std::memset(image.data() + padding, 0xff, offsetof(ChipCallable, storage_) - padding);
    ASSERT_TRUE(make_prepared_kernel_callable(resident, &prepared));
    expect_equal(prepared, before);
}

TEST_F(TmrKernelPreparedCallableTest, SparseTablePointsToCoreCallableDescriptorsNotInstructions) {
    ASSERT_EQ(prepared.functions.size(), 8u);
    const auto view = prepared.view();
    EXPECT_EQ(view.functions.entries, prepared.functions.data());
    EXPECT_EQ(view.functions.count, prepared.functions.size());
    EXPECT_EQ(view.identity.callable_id, resident.callable_id);
    for (int32_t id = 0; id < 8; ++id) {
        const CoreCallable *expected = id == 2 ? &child(0) : id == 7 ? &child(1) : nullptr;
        EXPECT_EQ(view.functions.lookup(id), reinterpret_cast<uint64_t>(expected));
        if (expected != nullptr) EXPECT_NE(view.functions.lookup(id), expected->resolved_addr());
    }
    EXPECT_EQ(view.functions.lookup(-1), 0u);
    EXPECT_EQ(view.functions.lookup(8), 0u);
    chip().child_func_ids_[1] = RUNTIME_MAX_FUNC_ID - 1;
    ASSERT_TRUE(make_prepared_kernel_callable(resident, &prepared));
    EXPECT_EQ(prepared.functions.size(), static_cast<size_t>(RUNTIME_MAX_FUNC_ID));
    EXPECT_EQ(prepared.view().functions.lookup(RUNTIME_MAX_FUNC_ID - 1), reinterpret_cast<uint64_t>(&child(1)));
}

TEST_F(TmrKernelPreparedCallableTest, IndependentCallablesWithSameFunctionIdsDoNotShareTables) {
    auto image_b = make_image();
    auto *chip_b = reinterpret_cast<ChipCallable *>(image_b.data());
    TmrCallableRegistrationArgs resident_b{27, reinterpret_cast<uint64_t>(image_b.data()), image_b.size(), 4, 0};
    PreparedKernelCallable prepared_b;
    ASSERT_TRUE(make_prepared_kernel_callable(resident_b, &prepared_b));
    EXPECT_NE(prepared.functions.data(), prepared_b.functions.data());
    for (int32_t i = 0; i < chip().child_count(); ++i) {
        const int32_t id = chip().child_func_id(i);
        EXPECT_EQ(prepared.view().functions.lookup(id), reinterpret_cast<uint64_t>(&chip().child(i)));
        EXPECT_EQ(prepared_b.view().functions.lookup(id), reinterpret_cast<uint64_t>(&chip_b->child(i)));
        EXPECT_NE(prepared.view().functions.lookup(id), prepared_b.view().functions.lookup(id));
    }
    const auto before = prepared;
    resident_b.context_generation = 28;
    ASSERT_TRUE(make_prepared_kernel_callable(resident_b, &prepared_b));
    expect_equal(prepared, before);
}

TEST_F(TmrKernelPreparedCallableTest, RejectsIdentityBoundsWithoutChangingExistingOutput) {
    EXPECT_FALSE(make_prepared_kernel_callable(resident, nullptr));
    for (int32_t id : {-1, MAX_REGISTERED_CALLABLE_IDS}) {
        auto invalid = resident;
        invalid.callable_id = id;
        reject(invalid);
    }
    auto invalid = resident;
    invalid.context_generation = 0;
    reject(invalid);
    invalid = resident;
    invalid.reserved = 1;
    reject(invalid);
    invalid = resident;
    invalid.device_address = 0;
    reject(invalid);
    invalid = resident;
    ++invalid.device_address;
    reject(invalid);
    invalid = resident;
    invalid.device_address = std::numeric_limits<uintptr_t>::max() - 7;
    reject(invalid);
    for (uint64_t bytes : {uint64_t{0}, uint64_t{sizeof(ChipCallable) - 1}, resident.bytes - 1, resident.bytes + 1}) {
        invalid = resident;
        invalid.bytes = bytes;
        reject(invalid);
    }
}

TEST_F(TmrKernelPreparedCallableTest, RejectsInvalidDuplicateAndUnresolvedChildMappingsTransactionally) {
    for (int32_t id : {-1, RUNTIME_MAX_FUNC_ID, std::numeric_limits<int32_t>::max()}) {
        chip().child_func_ids_[1] = id;
        reject();
    }
    chip().child_func_ids_[1] = chip().child_func_ids_[0];
    reject();
    chip().child_func_ids_[1] = 7;
    child(1).set_resolved_addr(0);
    reject();
}

TEST_F(TmrKernelPreparedCallableTest, RejectsMalformedSignaturesAndImageLayoutBeforeChildAccess) {
    const auto original = image;
    auto reset = [&] {
        std::memcpy(image.data(), original.data(), image.size());
    };
    for (int32_t count : {-1, CHIP_MAX_TENSOR_ARGS + 1}) {
        chip().sig_count_ = count;
        reject();
        reset();
    }
    chip().signature_[0] = static_cast<ArgDirection>(99);
    reject();
    reset();
    chip().signature_[0] = ArgDirection::SCALAR;
    reject();
    reset();
    for (int32_t count : {-1, 1025}) {
        chip().child_count_ = count;
        reject();
        reset();
    }
    for (uint32_t offset :
         {chip().child_offsets_[0], chip().child_offsets_[1] + 1, std::numeric_limits<uint32_t>::max()}) {
        chip().child_offsets_[1] = offset;
        reject();
        reset();
    }
    chip().binary_size_ = std::numeric_limits<uint32_t>::max();
    reject();
    reset();
    child(1).binary_size_ = std::numeric_limits<uint32_t>::max();
    reject();
    reset();
    child(1).sig_count_ = CORE_MAX_TENSOR_ARGS + 1;
    reject();
    reset();
    chip().func_name_len_ = CALLABLE_FUNC_NAME_MAX;
    reject();
    reset();
    chip().config_name_[chip().config_name_len_] = 'x';
    reject();
}

TEST_F(TmrKernelPreparedCallableTest, EmptyChildTableHasNoPhantomFunctionEntry) {
    auto empty = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch", nullptr, 0, nullptr, nullptr, 0, "config"
    );
    TmrCallableRegistrationArgs no_children{1, reinterpret_cast<uint64_t>(empty.data()), empty.size(), 0, 0};
    PreparedKernelCallable out;
    ASSERT_TRUE(make_prepared_kernel_callable(no_children, &out));
    EXPECT_EQ(out.identity.tensor_count, 0);
    EXPECT_EQ(out.identity.scalar_count, 0);
    EXPECT_TRUE(out.functions.empty());
    EXPECT_EQ(out.view().functions.lookup(0), 0u);
}
}  // namespace
