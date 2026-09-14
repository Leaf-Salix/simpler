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

#include <cstring>
#include <vector>

#include "task_interface/kernel_dispatch_args.h"
#include "task_interface/task_args.h"
#include "tensormap_and_ringbuffer/kernel_invocation.h"

namespace simpler::kernel {

// Storage is sized during callable preparation and leased exclusively until
// the transport has copied a submission. Encoding never grows the buffer.
class KernelDispatchPacket {
public:
    InvocationStatus prepare(const PreparedInvocationView &callable) {
        size_t bytes = 0;
        const auto rc = tmr::tmr_invocation_size(callable, &bytes);
        if (rc != InvocationStatus::Ok) return rc;
        bytes_.resize(offsetof(SimplerKernelDispatchArgs, invocation) + bytes);
        callable_ = callable;
        return InvocationStatus::Ok;
    }

    InvocationStatus encode(
        const ChipStorageTaskArgs &args, uint64_t residency, const tmr::TmrExecutionBindingView &binding,
        size_t sm_bytes, size_t arena_bytes
    ) noexcept {
        if (bytes_.empty() || residency == 0 || binding.device_binding_addr == 0 || binding.context_generation == 0 ||
            sm_bytes == 0 || arena_bytes == 0)
            return InvocationStatus::InvalidBinding;
        if (args.tensor_count() != callable_.tensor_count || args.scalar_count() != callable_.scalar_count)
            return InvocationStatus::InvalidCounts;
        SimplerKernelDispatchArgs prefix{};
        prefix.packet_bytes = bytes_.size();
        prefix.residency_address = residency;
        prefix.binding_address = binding.device_binding_addr;
        prefix.context_generation = binding.context_generation;
        prefix.sm_bytes = sm_bytes;
        prefix.arena_bytes = arena_bytes;
        prefix.invocation.mode = SIMPLER_MODE_KERNEL;
        prefix.invocation.callable_id = callable_.callable_id;
        prefix.invocation.generation = callable_.slot_generation;
        prefix.invocation.payload_bytes = bytes_.size() - sizeof(prefix);
        prefix.invocation.tensor_count = callable_.tensor_count;
        prefix.invocation.scalar_count = callable_.scalar_count;
        std::memcpy(bytes_.data(), &prefix, sizeof(prefix));
        const tmr::TmrBindingRef reference{binding.device_binding_addr, binding.context_generation};
        std::memcpy(bytes_.data() + sizeof(prefix), &reference, sizeof(reference));
        size_t offset = sizeof(prefix) + sizeof(reference);
        for (int32_t i = 0; i < callable_.tensor_count; ++i) {
            ChipTensor tensor{};
            const auto rc = tmr::normalize_invocation_tensor(args.tensor(i), &tensor);
            if (rc != InvocationStatus::Ok) return rc;
            std::memcpy(bytes_.data() + offset, &tensor, sizeof(tensor));
            offset += sizeof(tensor);
        }
        if (callable_.scalar_count != 0)
            std::memcpy(bytes_.data() + offset, args.scalar_data(), callable_.scalar_count * sizeof(uint64_t));
        return InvocationStatus::Ok;
    }

    ByteSpan packet() const noexcept { return {bytes_.data(), bytes_.size()}; }

private:
    PreparedInvocationView callable_{};
    std::vector<uint8_t> bytes_;
};

}  // namespace simpler::kernel
