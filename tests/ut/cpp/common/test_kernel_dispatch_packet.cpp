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
#include <vector>

#include "worker/kernel_dispatch_packet.h"

namespace {

using simpler::kernel::ByteSpan;
using simpler::kernel::InvocationStatus;
using simpler::kernel::KernelDispatchPacket;
using simpler::kernel::PreparedInvocationView;
using simpler::tmr::decode_tmr_invocation;
using simpler::tmr::TmrBindingRef;
using simpler::tmr::TmrExecutionBindingView;
using simpler::tmr::TmrInvocationView;

constexpr PreparedInvocationView kCallable{7, 1, 1, 19};
constexpr TmrExecutionBindingView kBinding{0x80000, 31};
constexpr uint64_t kResidency = 0x90000;
constexpr size_t kSmBytes = 0x100000;
constexpr size_t kArenaBytes = 0x200000;

ChipStorageTaskArgs make_args(uint64_t address = 0x10000, uint64_t scalar = 41) {
    ChipStorageTaskArgs args{};
    const uint32_t shape[] = {2, 4};
    args.add_tensor(
        make_tensor_external(reinterpret_cast<void *>(address), shape, 2, DataType::FLOAT32, AddressSpace::DEVICE)
    );
    args.add_scalar(scalar);
    return args;
}

ByteSpan invocation(ByteSpan packet) {
    constexpr size_t offset = offsetof(SimplerKernelDispatchArgs, invocation);
    return {packet.data + offset, packet.size - offset};
}

void expect_values(ByteSpan packet, uint64_t address, uint64_t scalar) {
    TmrInvocationView decoded;
    ASSERT_EQ(decode_tmr_invocation(invocation(packet), kCallable, kBinding, &decoded), InvocationStatus::Ok);
    ASSERT_EQ(decoded.tensor_count(), 1);
    ASSERT_EQ(decoded.scalar_count(), 1);
    ChipTensor tensor{};
    uint64_t value = 0;
    ASSERT_TRUE(decoded.tensor(0, &tensor));
    ASSERT_TRUE(decoded.scalar(0, &value));
    EXPECT_EQ(tensor.buffer.addr, address);
    EXPECT_EQ(tensor.buffer.size, 8 * sizeof(float));
    EXPECT_EQ(tensor.ndims, 2u);
    EXPECT_EQ(tensor.shapes[0], 2u);
    EXPECT_EQ(tensor.shapes[1], 4u);
    EXPECT_EQ(value, scalar);
}

TEST(KernelDispatchPacket, PrepareReservesExactWireSizeIncludingEmptyAndMaximumSignatures) {
    for (const auto &callable :
         {PreparedInvocationView{0, 0, 0, 1}, PreparedInvocationView{2, 3, 2, 9},
          PreparedInvocationView{7, CHIP_MAX_TENSOR_ARGS - CHIP_MAX_SCALAR_ARGS, CHIP_MAX_SCALAR_ARGS, 19}}) {
        KernelDispatchPacket packet;
        ASSERT_EQ(packet.prepare(callable), InvocationStatus::Ok);
        EXPECT_EQ(
            packet.packet().size, sizeof(SimplerKernelDispatchArgs) + sizeof(TmrBindingRef) +
                                      callable.tensor_count * sizeof(ChipTensor) +
                                      callable.scalar_count * sizeof(uint64_t)
        );
    }
    KernelDispatchPacket empty;
    const PreparedInvocationView callable{0, 0, 0, 1};
    ASSERT_EQ(empty.prepare(callable), InvocationStatus::Ok);
    ASSERT_EQ(empty.encode({}, kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::Ok);
    TmrInvocationView decoded;
    ASSERT_EQ(decode_tmr_invocation(invocation(empty.packet()), callable, kBinding, &decoded), InvocationStatus::Ok);
    EXPECT_EQ(decoded.tensor_count(), 0);
    EXPECT_EQ(decoded.scalar_count(), 0);
}

TEST(KernelDispatchPacket, EncodedDispatchEnvelopeMatchesTheRuntimeDecoder) {
    KernelDispatchPacket packet;
    ASSERT_EQ(packet.prepare(kCallable), InvocationStatus::Ok);
    ASSERT_EQ(packet.encode(make_args(), kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::Ok);
    SimplerKernelDispatchArgs envelope{};
    std::memcpy(&envelope, packet.packet().data, sizeof(envelope));
    EXPECT_EQ(envelope.packet_bytes, packet.packet().size);
    EXPECT_EQ(envelope.residency_address, kResidency);
    EXPECT_EQ(envelope.binding_address, kBinding.device_binding_addr);
    EXPECT_EQ(envelope.context_generation, kBinding.context_generation);
    EXPECT_EQ(envelope.sm_bytes, kSmBytes);
    EXPECT_EQ(envelope.arena_bytes, kArenaBytes);
    EXPECT_EQ(envelope.invocation.callable_id, kCallable.callable_id);
    EXPECT_EQ(envelope.invocation.generation, kCallable.slot_generation);
    EXPECT_EQ(envelope.invocation.payload_bytes, packet.packet().size - sizeof(envelope));
    EXPECT_EQ(envelope.invocation.host_copy_tensor_count, 0);
    expect_values(packet.packet(), 0x10000, 41);

    TmrInvocationView decoded;
    auto stale = kCallable;
    ++stale.slot_generation;
    EXPECT_EQ(
        decode_tmr_invocation(invocation(packet.packet()), stale, kBinding, &decoded), InvocationStatus::StaleCallable
    );
    auto different_context = kBinding;
    ++different_context.context_generation;
    EXPECT_EQ(
        decode_tmr_invocation(invocation(packet.packet()), kCallable, different_context, &decoded),
        InvocationStatus::InvalidBinding
    );
}

TEST(KernelDispatchPacket, ReusesPreparedStorageWhileTransportSnapshotsKeepTheirOwnArguments) {
    KernelDispatchPacket packet;
    ASSERT_EQ(packet.prepare(kCallable), InvocationStatus::Ok);
    const auto storage = packet.packet();
    auto args = make_args();
    ASSERT_EQ(packet.encode(args, kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::Ok);
    const std::vector<uint8_t> first(storage.data, storage.data + storage.size);
    args.clear();
    expect_values(packet.packet(), 0x10000, 41);

    for (uint64_t i = 1; i <= 64; ++i) {
        args = make_args(0x20000 + i * 64, 1000 + i);
        ASSERT_EQ(packet.encode(args, kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::Ok);
        EXPECT_EQ(packet.packet().data, storage.data);
        EXPECT_EQ(packet.packet().size, storage.size);
        args.tensor(0).buffer.addr = 0xdeadbeef;
        args.scalar(0) = 0;
        expect_values(packet.packet(), 0x20000 + i * 64, 1000 + i);
    }
    expect_values({first.data(), first.size()}, 0x10000, 41);
}

TEST(KernelDispatchPacket, RejectsBadPreparationWithoutLosingThePreparedPacket) {
    KernelDispatchPacket packet;
    ASSERT_EQ(packet.prepare(kCallable), InvocationStatus::Ok);
    const auto storage = packet.packet();
    for (const auto &bad :
         {PreparedInvocationView{-1, 1, 1, 1}, PreparedInvocationView{0, -1, 1, 1}, PreparedInvocationView{0, 1, -1, 1},
          PreparedInvocationView{0, 1, 1, 0}, PreparedInvocationView{0, CHIP_MAX_TENSOR_ARGS, 1, 1},
          PreparedInvocationView{0, 0, CHIP_MAX_SCALAR_ARGS + 1, 1}}) {
        EXPECT_EQ(packet.prepare(bad), InvocationStatus::InvalidArgument);
        EXPECT_EQ(packet.packet().data, storage.data);
        EXPECT_EQ(packet.packet().size, storage.size);
    }
    ASSERT_EQ(packet.encode(make_args(), kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::Ok);
    expect_values(packet.packet(), 0x10000, 41);
}

TEST(KernelDispatchPacket, RejectsInvalidCountsBeforeReadingArgumentStorage) {
    KernelDispatchPacket packet;
    ASSERT_EQ(packet.prepare(kCallable), InvocationStatus::Ok);
    for (int32_t count : {-1, 0, 2, CHIP_MAX_TENSOR_ARGS + 1}) {
        auto args = make_args();
        args.tensor_count_ = count;
        EXPECT_EQ(packet.encode(args, kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::InvalidCounts);
        args = make_args();
        args.scalar_count_ = count;
        EXPECT_EQ(packet.encode(args, kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::InvalidCounts);
    }
}

TEST(KernelDispatchPacket, RejectsInvalidTensorViewsAndCanEncodeTheNextValidInvocation) {
    KernelDispatchPacket packet;
    ASSERT_EQ(packet.prepare(kCallable), InvocationStatus::Ok);
    using Mutate = void (*)(ChipTensor &);
    const Mutate mutations[] = {
        [](auto &t) {
            t.buffer.addr = 0;
        },
        [](auto &t) {
            t.address_space = AddressSpace::HOST;
        },
        [](auto &t) {
            t.ndims = 0;
        },
        [](auto &t) {
            t.ndims = MAX_TENSOR_DIMS + 1;
        },
        [](auto &t) {
            t.buffer.size = sizeof(float);
        },
        [](auto &t) {
            t.strides[0] = 0;
        },
        [](auto &t) {
            t.start_offset = std::numeric_limits<uint64_t>::max();
        },
    };
    const auto storage = packet.packet();
    for (auto mutate : mutations) {
        auto args = make_args();
        mutate(args.tensor(0));
        EXPECT_EQ(packet.encode(args, kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::InvalidTensor);
        EXPECT_EQ(packet.packet().data, storage.data);
        EXPECT_EQ(packet.packet().size, storage.size);
    }
    ASSERT_EQ(packet.encode(make_args(0x40000, 91), kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::Ok);
    expect_values(packet.packet(), 0x40000, 91);
}

TEST(KernelDispatchPacket, RejectsMissingResidencyContextAndExecutionExtents) {
    KernelDispatchPacket packet;
    auto args = make_args();
    EXPECT_EQ(packet.encode(args, kResidency, kBinding, kSmBytes, kArenaBytes), InvocationStatus::InvalidBinding);
    ASSERT_EQ(packet.prepare(kCallable), InvocationStatus::Ok);
    EXPECT_EQ(packet.encode(args, 0, kBinding, kSmBytes, kArenaBytes), InvocationStatus::InvalidBinding);
    EXPECT_EQ(packet.encode(args, kResidency, {0, 31}, kSmBytes, kArenaBytes), InvocationStatus::InvalidBinding);
    EXPECT_EQ(packet.encode(args, kResidency, {0x80000, 0}, kSmBytes, kArenaBytes), InvocationStatus::InvalidBinding);
    EXPECT_EQ(packet.encode(args, kResidency, kBinding, 0, kArenaBytes), InvocationStatus::InvalidBinding);
    EXPECT_EQ(packet.encode(args, kResidency, kBinding, kSmBytes, 0), InvocationStatus::InvalidBinding);
}

}  // namespace
