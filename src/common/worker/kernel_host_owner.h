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

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "call_config.h"
#include "runtime_c_api.h"
#include "task_args.h"

namespace simpler::kernel {

class KernelQueuedInvocation;

// Owns only the Host right to call a kernel context, not its Device resources.
// All users of the ctx must go through this owner. Device/graph quiescence
// before close is an external precondition, independent of pending callbacks.
class KernelHostOwner : public std::enable_shared_from_this<KernelHostOwner> {
public:
    explicit KernelHostOwner(const std::string &runtime_path);
    ~KernelHostOwner();
    KernelHostOwner(const KernelHostOwner &) = delete;
    KernelHostOwner &operator=(const KernelHostOwner &) = delete;

    int initialize(
        int device, const std::vector<uint8_t> &cpu, const std::vector<uint8_t> &core,
        const std::vector<uint8_t> &dispatcher, const CallConfig &config, uint64_t generation
    );
    int prepare(int32_t cid, const void *callable, size_t bytes);
    int launch(int32_t cid, const ChipStorageTaskArgs *args, void *stream);
    int close();
    bool ready() const;
    bool closed() const;
    int device() const;
    size_t pending() const;
    int supported() const;
    size_t committed_device_memory() const;
    std::shared_ptr<KernelQueuedInvocation>
    accept(int32_t cid, const ChipTensor *tensors, size_t tensor_count, const uint64_t *scalars, size_t scalar_count);

private:
    friend class KernelQueuedInvocation;
    mutable std::mutex gate_;
    void *library_{nullptr};
    DeviceContextHandle ctx_{nullptr};
    decltype(&destroy_device_context) destroy_{nullptr};
    decltype(&finalize_device) finalize_{nullptr};
    decltype(&simpler_kernel_mode_init) init_{nullptr};
    decltype(&simpler_kernel_mode_prepare_callable) prepare_{nullptr};
    decltype(&simpler_kernel_mode_launch) launch_{nullptr};
    decltype(&simpler_kernel_mode_supported) supported_{nullptr};
    decltype(&committed_device_memory_ctx) committed_memory_{nullptr};
    enum class State { Created, Ready, Closing, Closed };
    State state_{State::Created};
    int device_{-1};
    size_t pending_{0};
};

class KernelQueuedInvocation {
public:
    ~KernelQueuedInvocation();
    KernelQueuedInvocation(const KernelQueuedInvocation &) = delete;
    KernelQueuedInvocation &operator=(const KernelQueuedInvocation &) = delete;
    // The hook runs once under the same gate, before native enqueue.
    // It must not reenter the owner. No Python object is accessed here.
    int invoke(void *stream, int (*before)(void *) noexcept = nullptr, void *data = nullptr) noexcept;

private:
    friend class KernelHostOwner;
    KernelQueuedInvocation(
        std::shared_ptr<KernelHostOwner> owner, int32_t cid, const ChipTensor *tensors, size_t tensor_count,
        const uint64_t *scalars, size_t scalar_count
    );
    std::shared_ptr<KernelHostOwner> owner_;
    int32_t cid_;
    std::vector<ChipTensor> tensors_;
    std::vector<uint64_t> scalars_;
    bool admitted_{false};
    bool done_{false};
    int result_{PTO_RUNTIME_ERR_INVALID_STATE};
};

}  // namespace simpler::kernel
