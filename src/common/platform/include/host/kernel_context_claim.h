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
#include <mutex>
#include <set>
#include <utility>

#include "worker/runtime_c_api.h"

// One registry belongs to each loaded host runtime SO. Its keys match the
// device loader's resident runtime identity: device id and inner SO build id.
class KernelContextClaimRegistry {
    friend class KernelContextClaim;
    std::mutex mutex_;
    std::set<std::pair<int, uint64_t>> owners_;
};

class KernelContextClaim {
public:
    KernelContextClaim() = default;
    KernelContextClaim(const KernelContextClaim &) = delete;
    KernelContextClaim &operator=(const KernelContextClaim &) = delete;
    // Destruction is not proof that device work has stopped. Only explicit
    // successful finalize releases a live claim; failed close stays exclusive.
    ~KernelContextClaim() = default;

    int acquire(KernelContextClaimRegistry &registry, int device_id, uint64_t runtime_fingerprint) {
        if (registry_ != nullptr || device_id < 0) return PTO_RUNTIME_ERR_INVALID_STATE;
        std::lock_guard<std::mutex> lock(registry.mutex_);
        const std::pair<int, uint64_t> key{device_id, runtime_fingerprint};
        if (!registry.owners_.insert(key).second) return PTO_RUNTIME_ERR_INVALID_STATE;
        key_ = key;
        registry_ = &registry;
        return 0;
    }

    bool held() const { return registry_ != nullptr; }

    void rollback_initialization() { release(); }
    void finish_finalize(int status) {
        if (status == 0) release();
    }

private:
    void release() {
        if (registry_ == nullptr) return;
        std::lock_guard<std::mutex> lock(registry_->mutex_);
        registry_->owners_.erase(key_);
        registry_ = nullptr;
    }

    KernelContextClaimRegistry *registry_{nullptr};
    std::pair<int, uint64_t> key_{};
};
