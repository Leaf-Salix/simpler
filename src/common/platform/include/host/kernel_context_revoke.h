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

#include "worker/runtime_c_api.h"

// Close-only operations over prepare-owned receipt/event storage. Submission
// success and completion are separate; completion waits are bounded and cover
// only the owner's control stream. The submission mutex serializes every operation.
struct KernelRevokeOps {
    void *context{nullptr};
    int (*enqueue_revoke)(void *) noexcept {nullptr};
    int (*enqueue_receipt_copy)(void *) noexcept {nullptr};
    int (*record_completion)(void *) noexcept {nullptr};
    int (*wait_completion)(void *) noexcept {nullptr};
    int (*validate_receipt)(void *) noexcept {nullptr};

    bool valid() const {
        return enqueue_revoke != nullptr && enqueue_receipt_copy != nullptr && record_completion != nullptr &&
               wait_completion != nullptr && validate_receipt != nullptr;
    }
};

// Submission ledger only, not a resource owner. The context keeps all device
// allocations, pinned Host receipt and completion event alive until advance()
// succeeds. The caller first closes launch admission; external graph quiescence
// and destruction remain required because capture's original hidden streams
// do not prove that later graph replay has stopped using the allocations.
class KernelContextRevoke {
public:
    void registration_may_exist() noexcept { required_ = true; }
    bool confirmed() const noexcept { return !required_ || stage_ == Stage::Confirmed; }

    int advance(const KernelRevokeOps &ops) noexcept {
        if (confirmed()) return 0;
        if (!ops.valid()) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (stage_ == Stage::New) {
            const int rc = ops.enqueue_revoke(ops.context);
            if (rc != 0) return rc;
            stage_ = Stage::RevokeQueued;
        }
        if (stage_ == Stage::RevokeQueued) {
            const int rc = ops.enqueue_receipt_copy(ops.context);
            if (rc != 0) return rc;
            stage_ = Stage::CopyQueued;
        }
        if (stage_ == Stage::CopyQueued) {
            const int rc = ops.record_completion(ops.context);
            if (rc != 0) return rc;
            stage_ = Stage::Recorded;
        }
        // An event that was never recorded may query COMPLETE. The stage
        // check above is therefore essential, not just a retry optimization.
        const int rc = ops.wait_completion(ops.context);
        if (rc != 0) return rc;
        const int receipt_rc = ops.validate_receipt(ops.context);
        if (receipt_rc != 0) return receipt_rc;
        stage_ = Stage::Confirmed;
        return 0;
    }

private:
    enum class Stage { New, RevokeQueued, CopyQueued, Recorded, Confirmed };
    Stage stage_{Stage::New};
    bool required_{false};
};
