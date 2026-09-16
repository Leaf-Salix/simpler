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
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "tmr_kernel_context.h"

namespace simpler::tmr {

enum class TmrRevokeCompletion : uint32_t { Pending = 0, Complete = 1 };

// Prepare owns this isolated device line and initializes its identity, with
// status/complete/reserved zero. It remains pinned through the ordered D2H copy
// and a successfully recorded completion event. Each context uses it once.
struct alignas(64) TmrContextRevokeReceipt {
    uint64_t descriptor_address;
    uint64_t context_generation;
    int32_t status;
    uint32_t complete;
    uint64_t reserved[5];
};

// Internal, exclusive Host-owner transport. The owner supplies a writable,
// prepare-owned receipt, disjoint from every live borrowed region.
// Address checks do not authenticate arbitrary forged physical addresses.
struct TmrContextRevokeArgs {
    uint64_t descriptor_address;
    uint64_t context_generation;
    uint64_t receipt_address;
    uint64_t receipt_bytes;
};

inline bool valid_tmr_context_revoke_args(const TmrContextRevokeArgs &args) noexcept {
    return args.descriptor_address != 0 && args.context_generation != 0 &&
           args.descriptor_address % alignof(TmrKernelContextDescriptor) == 0 &&
           args.descriptor_address <= UINTPTR_MAX - sizeof(TmrKernelContextDescriptor) && args.receipt_address != 0 &&
           args.receipt_address % alignof(TmrContextRevokeReceipt) == 0 &&
           args.receipt_bytes == sizeof(TmrContextRevokeReceipt) &&
           args.receipt_address <= UINTPTR_MAX - sizeof(TmrContextRevokeReceipt) &&
           (args.receipt_address + sizeof(TmrContextRevokeReceipt) <= args.descriptor_address ||
            args.descriptor_address + sizeof(TmrKernelContextDescriptor) <= args.receipt_address);
}

// Host checks Complete only after the receipt copy's recorded event completes;
// neither native enqueue success nor an unrecorded event validates this value.
inline bool valid_tmr_context_revoke_receipt(
    const TmrContextRevokeReceipt &receipt, const TmrContextRevokeArgs &args, TmrRevokeCompletion completion
) noexcept {
    if (!valid_tmr_context_revoke_args(args) || receipt.descriptor_address != args.descriptor_address ||
        receipt.context_generation != args.context_generation || receipt.status != 0 ||
        receipt.complete != static_cast<uint32_t>(completion))
        return false;
    for (uint64_t reserved : receipt.reserved)
        if (reserved != 0) return false;
    return true;
}

static_assert(
    std::is_trivially_copyable_v<TmrContextRevokeReceipt> && std::is_standard_layout_v<TmrContextRevokeReceipt>
);
static_assert(sizeof(TmrContextRevokeReceipt) == 64 && alignof(TmrContextRevokeReceipt) == 64);
static_assert(offsetof(TmrContextRevokeReceipt, descriptor_address) == 0);
static_assert(offsetof(TmrContextRevokeReceipt, context_generation) == 8);
static_assert(offsetof(TmrContextRevokeReceipt, status) == 16);
static_assert(offsetof(TmrContextRevokeReceipt, complete) == 20);
static_assert(offsetof(TmrContextRevokeReceipt, reserved) == 24);
static_assert(std::is_trivially_copyable_v<TmrContextRevokeArgs> && std::is_standard_layout_v<TmrContextRevokeArgs>);
static_assert(sizeof(TmrContextRevokeArgs) == 32 && alignof(TmrContextRevokeArgs) == 8);
static_assert(offsetof(TmrContextRevokeArgs, descriptor_address) == 0);
static_assert(offsetof(TmrContextRevokeArgs, context_generation) == 8);
static_assert(offsetof(TmrContextRevokeArgs, receipt_address) == 16);
static_assert(offsetof(TmrContextRevokeArgs, receipt_bytes) == 24);

}  // namespace simpler::tmr
