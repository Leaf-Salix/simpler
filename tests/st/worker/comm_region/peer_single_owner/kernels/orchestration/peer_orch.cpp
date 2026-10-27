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

#include <stdint.h>
#include <string.h>

#include "aicpu/worker_chip_orch_endpoint.h"
#include "orchestration_api.h"

namespace {
constexpr uint32_t kWords = 128;
constexpr uint64_t kBytes = kWords * sizeof(uint32_t);

bool check_endpoint(const WorkerChipOrchEndpoint &endpoint, bool success) {
    if (!success) {
        rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "peer endpoint failure: %s", endpoint.error().message);
    }
    return success;
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 9};
}

__attribute__((visibility("default"))) void peer_orchestration(const ChipTaskArgs &args) {
    uint64_t binding[WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT] = {
        args.scalar(0), args.scalar(1), args.scalar(2), args.scalar(3), args.scalar(4), args.scalar(5),
    };
    WorkerChipOrchEndpoint endpoint(binding, WORKER_CHIP_ORCH_REGION_DESC_SCALAR_COUNT);
    const uint32_t phase = static_cast<uint32_t>(args.scalar(6));
    const uint32_t seed = static_cast<uint32_t>(args.scalar(7));
    uint64_t ready = 0;
    uint64_t done = 0;
    if (!check_endpoint(endpoint, endpoint.counter_addr(0, ready) && endpoint.counter_addr(64, done))) {
        return;
    }
    uint32_t values[kWords];
    if (phase == 0) {
        for (uint32_t i = 0; i < kWords; ++i) {
            values[i] = seed + i;
        }
    } else {
        int32_t observed = 0;
        if (!check_endpoint(
                endpoint, endpoint.signal_wait(
                              phase == 1 ? ready : done, static_cast<int32_t>(seed), WorkerChipOrchWaitCmp::EQ,
                              5000000000ULL, observed
                          )
            )) {
            return;
        }
        WorkerChipOrchPayloadView view{};
        if (!check_endpoint(endpoint, endpoint.payload_read(0, kBytes, view))) {
            return;
        }
        const uint32_t *shared = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(view.gm_addr));
        for (uint32_t i = 0; i < kWords; ++i) {
            uint32_t expected = phase == 1 ? seed + i : (seed + i) ^ 0xFFFF;
            if (shared[i] != expected) {
                rt_report_fatal(
                    SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "peer payload mismatch phase=%u index=%u actual=%u expected=%u",
                    phase, i, shared[i], expected
                );
                return;
            }
            values[i] = (seed + i) ^ 0xFFFF;
        }
    }
    if (phase < 2 &&
        !check_endpoint(
            endpoint, endpoint.payload_write(0, values, kBytes) &&
                          endpoint.signal_notify(
                              phase == 0 ? ready : done, static_cast<int32_t>(seed), WorkerChipOrchNotifyOp::Set
                          )
        )) {
        return;
    }
    void *result = args.tensor(0).ref().data_as<void>();
    memcpy(result, values, kBytes);
    cache_flush_range(result, kBytes);
}
}  // extern "C"
