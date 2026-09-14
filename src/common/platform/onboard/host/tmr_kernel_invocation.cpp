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

#include "tmr_kernel_invocation.h"

#include <cstring>
#include <vector>

#include "kernel_dispatch_args.h"
#include "device_runner_base.h"

namespace simpler::tmr {

int enqueue_tmr_invocation_aicpu(
    DeviceRunnerBase &runner, void *stream, int32_t aicpu_num, const TmrEncodingCandidate &candidate,
    const PreparedInvocationView &callable, const TmrExecutionBindingView &binding
) noexcept {
    if (stream == nullptr || aicpu_num <= 0) return PTO_RUNTIME_ERR_INTERNAL;
    const auto status = validate_tmr_submission(candidate, callable, binding);
    if (status == InvocationStatus::StaleCallable || status == InvocationStatus::InvalidBinding)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (status != InvocationStatus::Ok) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        const auto packet = candidate.packet();
        if (packet.size < sizeof(SimplerKernelInvocationHeader)) return PTO_RUNTIME_ERR_INTERNAL;
        // The public K9 header remains the only invocation identity. The outer
        // dispatch prefix adds the trusted residency descriptor address; its
        // payload is the K9 payload (binding ref + descriptors + scalars), not
        // a second copy of the K9 header.
        SimplerKernelDispatchArgs dispatch{};
        std::memcpy(&dispatch.invocation, packet.data, sizeof(dispatch.invocation));
        dispatch.residency_address = binding.device_binding_addr;
        dispatch.round_epoch = runner.next_kernel_round_epoch();
        dispatch.invocation.payload_bytes = packet.size - sizeof(SimplerKernelInvocationHeader);
        dispatch.packet_bytes = sizeof(dispatch) + dispatch.invocation.payload_bytes;
        std::vector<uint8_t> launch_packet(static_cast<size_t>(dispatch.packet_bytes));
        std::memcpy(launch_packet.data(), &dispatch, sizeof(dispatch));
        std::memcpy(
            launch_packet.data() + sizeof(dispatch), packet.data + sizeof(SimplerKernelInvocationHeader),
            static_cast<size_t>(dispatch.invocation.payload_bytes)
        );
        return runner.launch_aicpu_payload(
            stream, launch_packet.data(), launch_packet.size(), TmrKernelInvocationName, aicpu_num
        );
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

}  // namespace simpler::tmr
