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
#include <limits>

#include "kernel_dispatch_args.h"
#include "callable_protocol.h"
#include "chip_callable_layout.h"
#include "aicpu/kernel_invocation_consumer.h"

namespace {
int consumed;
int consumer_status;
const void *seen_payload;
int32_t seen_id;
const ChipCallable *seen_callable;
size_t seen_callable_bytes;
uint64_t seen_function;
int32_t seen_function_id;
uint64_t seen_binding;
uint64_t seen_context_generation;
uint64_t seen_sm_bytes;
uint64_t seen_arena_bytes;
struct Packet {
    SimplerKernelDispatchArgs args;
    uint64_t payload;
};
class KernelDispatch : public testing::Test {
protected:
    Packet packet{};
    std::vector<uint8_t> image;
    void SetUp() override {
        consumed = 0;
        consumer_status = -83;
        seen_payload = nullptr;
        seen_binding = seen_context_generation = seen_sm_bytes = seen_arena_bytes = 0;
        const uint8_t code[] = {1, 2, 3};
        const int32_t func_id = 7;
        auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, code, sizeof(code));
        image = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
            nullptr, 0, "orch", code, sizeof(code), &func_id, &child, 1, ""
        );
        const auto *callable = reinterpret_cast<const ChipCallable *>(image.data());
        patch_chip_callable_scratch_for_device(
            callable, compute_chip_callable_layout(callable), reinterpret_cast<uint64_t>(image.data()), image.data()
        );
        packet.args.chip_callable_address = reinterpret_cast<uint64_t>(image.data());
        packet.args.chip_callable_bytes = image.size();
        packet.args.packet_bytes = sizeof(packet);
        packet.args.invocation.mode = SIMPLER_MODE_KERNEL;
        packet.args.invocation.callable_id = 63;
        packet.args.invocation.payload_bytes = sizeof(packet.payload);
        packet.payload = 42;
    }
    int run() { return simpler_aicpu_kernel_exec(&packet); }
    void invalid() {
        EXPECT_EQ(run(), 2);
        EXPECT_EQ(consumed, 0);
    }
};
}  // namespace

// The exported production entry forwards to this test consumer.
int consume_kernel_invocation(
    const SimplerKernelDispatchArgs &args, const ChipCallable &callable, size_t callable_bytes, const void *payload,
    size_t bytes
) {
    const auto &invocation = args.invocation;
    ++consumed;
    seen_binding = args.binding_address;
    seen_context_generation = args.context_generation;
    seen_sm_bytes = args.sm_bytes;
    seen_arena_bytes = args.arena_bytes;
    seen_callable = &callable;
    seen_callable_bytes = callable_bytes;
    seen_function_id = callable.child_func_ids_[0];
    seen_function = callable.child(0).resolved_addr();
    seen_payload = payload;
    seen_id = invocation.callable_id;
    EXPECT_EQ(bytes, sizeof(uint64_t));
    EXPECT_EQ(*static_cast<const uint64_t *>(payload), 42);
    return consumer_status;
}

TEST_F(KernelDispatch, ValidSnapshotReachesConsumerAndNormalizesNativeResult) {
    const Packet before = packet;
    const auto original_image = image;
    EXPECT_EQ(run(), 2);
    EXPECT_EQ(run(), 2);
    EXPECT_EQ(consumed, 2);
    EXPECT_EQ(seen_payload, &packet.payload);
    EXPECT_EQ(seen_id, 63);
    EXPECT_EQ(seen_callable, reinterpret_cast<const ChipCallable *>(image.data()));
    EXPECT_EQ(seen_callable_bytes, image.size());
    EXPECT_EQ(seen_function_id, 7);
    EXPECT_EQ(
        seen_function, packet.args.chip_callable_address + offsetof(ChipCallable, storage_) +
                           seen_callable->child_offset(0) + CoreCallable::binary_data_offset()
    );
    EXPECT_EQ(image, original_image);
    EXPECT_EQ(std::memcmp(&packet, &before, sizeof(packet)), 0);
}
TEST_F(KernelDispatch, NativeBoundaryAcceptsOnlySuccessOrInnerError) {
    for (int result : {0, 1, 2, 4, 6, 7, -83, std::numeric_limits<int>::max()}) {
        consumer_status = result;
        EXPECT_EQ(run(), result == 0 ? 0 : 2);
    }
    EXPECT_EQ(consumed, 8);
}
TEST_F(KernelDispatch, RejectsInvalidIdentityBeforeConsumingPayload) {
    for (int id : {-1, MAX_REGISTERED_CALLABLE_IDS, std::numeric_limits<int32_t>::max()}) {
        packet.args.invocation.callable_id = id;
        invalid();
    }
    packet.args.invocation.callable_id = 0;
    packet.args.invocation.mode = SIMPLER_MODE_PROGRAM;
    invalid();
}
TEST_F(KernelDispatch, RejectsMalformedEnvelope) {
    EXPECT_EQ(simpler_aicpu_kernel_exec(nullptr), 2);
    EXPECT_EQ(simpler_aicpu_kernel_exec(reinterpret_cast<char *>(&packet) + 1), 2);
    for (uint64_t bytes :
         {uint64_t(0), uint64_t(sizeof(SimplerKernelDispatchArgs) - 1), std::numeric_limits<uint64_t>::max()}) {
        packet.args.packet_bytes = bytes;
        invalid();
    }
    packet.args.packet_bytes = sizeof(packet);
    packet.args.invocation.payload_bytes++;
    invalid();
}
TEST_F(KernelDispatch, RejectsInvalidCountsAndReservedFields) {
    const auto valid = packet.args.invocation;
    for (int count : {-1, 257}) {
        packet.args.invocation.tensor_count = count;
        invalid();
    }
    packet.args.invocation = valid;
    for (int count : {-1, 129}) {
        packet.args.invocation.scalar_count = count;
        invalid();
    }
    packet.args.invocation.tensor_count = 256;
    packet.args.invocation.scalar_count = 1;
    invalid();
    packet.args.invocation = valid;
    packet.args.invocation.host_copy_tensor_count = 1;
    invalid();
    packet.args.invocation = valid;
    packet.args.invocation.reserved_ = 1;
    invalid();
}

TEST_F(KernelDispatch, RejectsInvalidCallableSpanBeforeConsumer) {
    const auto address = packet.args.chip_callable_address;
    for (uint64_t bad : {uint64_t(0), address + 1, std::numeric_limits<uint64_t>::max() - 15}) {
        packet.args.chip_callable_address = bad;
        invalid();
    }
    packet.args.chip_callable_address = address;
    packet.args.chip_callable_bytes = sizeof(ChipCallable) - 1;
    invalid();
    packet.args.chip_callable_bytes = std::numeric_limits<uint64_t>::max();
    invalid();
}

TEST_F(KernelDispatch, EachInvocationReceivesItsOwnCallableFunctionMapping) {
    EXPECT_EQ(run(), 2);
    const auto first_function = seen_function;
    auto other = image;
    auto *callable = reinterpret_cast<ChipCallable *>(other.data());
    callable->child_func_ids_[0] = 19;
    patch_chip_callable_scratch_for_device(
        callable, compute_chip_callable_layout(callable), reinterpret_cast<uint64_t>(other.data()), other.data()
    );
    packet.args.chip_callable_address = reinterpret_cast<uint64_t>(other.data());
    packet.args.invocation.callable_id = 3;
    EXPECT_EQ(run(), 2);
    EXPECT_EQ(seen_id, 3);
    EXPECT_EQ(seen_function_id, 19);
    EXPECT_NE(seen_function, first_function);
    EXPECT_EQ(seen_callable, callable);
    packet.args.chip_callable_address = reinterpret_cast<uint64_t>(image.data());
    packet.args.invocation.callable_id = 63;
    EXPECT_EQ(run(), 2);
    EXPECT_EQ(seen_function_id, 7);
    EXPECT_EQ(seen_function, first_function);
}

TEST_F(KernelDispatch, ForwardsContextBindingAndRegionExtentsToRuntimeConsumer) {
    packet.args.binding_address = 0x200000;
    packet.args.context_generation = 83;
    packet.args.sm_bytes = 4096;
    packet.args.arena_bytes = 8192;
    EXPECT_EQ(run(), 2);
    EXPECT_EQ(seen_binding, 0x200000u);
    EXPECT_EQ(seen_context_generation, 83u);
    EXPECT_EQ(seen_sm_bytes, 4096u);
    EXPECT_EQ(seen_arena_bytes, 8192u);
}
