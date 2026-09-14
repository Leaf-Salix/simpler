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

#include <cstdint>

/**
 * Kernel-only fork/join submission protocol.
 *
 * The binder owns no stream, event, callable, or device allocation. The
 * caller supplies operations for one invocation. This keeps program mode
 * completely outside the protocol while making the ordering and failure
 * policy testable without a CANN dependency.
 */
class KernelBinder {
public:
    struct Operations {
        using Step = int (*)(void *) noexcept;
        using Error = void (*)(void *, int) noexcept;
        void *context{nullptr};
        Step clear_round{nullptr};
        Step record_start{nullptr};
        Step wait_aicore_start{nullptr};
        Step launch_aicore{nullptr};
        Step record_aicore_done{nullptr};
        Step wait_aicpu_start{nullptr};
        Step launch_aicpu{nullptr};
        Step record_aicpu_done{nullptr};
        Step join_aicpu{nullptr};
        Step join_aicore{nullptr};
        Step record_tail{nullptr};
        Error poison{nullptr};
        Error cancel{nullptr};
    };

    KernelBinder() = default;
    KernelBinder(const KernelBinder &) = delete;
    KernelBinder &operator=(const KernelBinder &) = delete;

    int submit(const Operations &ops) const noexcept {
        const auto step = [&](Operations::Step operation) noexcept {
            return operation == nullptr ? -1 : operation(ops.context);
        };
        int rc = step(ops.clear_round);
        if (rc == 0) rc = step(ops.record_start);
        if (rc == 0) rc = step(ops.wait_aicore_start);
        if (rc == 0) rc = step(ops.launch_aicore);
        if (rc == 0) rc = step(ops.record_aicore_done);
        if (rc == 0) rc = step(ops.wait_aicpu_start);
        if (rc == 0) rc = step(ops.launch_aicpu);
        if (rc == 0) rc = step(ops.record_aicpu_done);
        if (rc == 0) rc = step(ops.join_aicpu);
        if (rc == 0) rc = step(ops.join_aicore);
        if (rc == 0) rc = step(ops.record_tail);
        if (rc != 0) {
            // cancel is intentionally optional: until K7 supplies a real
            // device-side cancellation protocol, poison remains the only
            // safe fallback and the caller must not reuse the context.
            if (ops.cancel != nullptr) ops.cancel(ops.context, rc);
            if (ops.poison != nullptr) ops.poison(ops.context, rc);
        }
        return rc;
    }
};
