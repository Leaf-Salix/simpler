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

#include "kernel_host_owner.h"

#include <cstdio>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>

#include "common/host_log_binding.h"
#include "host_log.h"
#include "kernel_invocation_validation.h"

namespace simpler::kernel {
namespace {
template <typename T>
T symbol(void *library, const char *name) {
    dlerror();
    auto ptr = dlsym(library, name);
    if (const char *error = dlerror()) throw std::runtime_error(std::string(name) + ": " + error);
    return reinterpret_cast<T>(ptr);
}
}  // namespace

KernelHostOwner::KernelHostOwner(const std::string &path) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(std::string("kernel runtime dlopen: ") + dlerror());
    try {
        const char *error = nullptr;
        if (simpler::log::bind_loaded_host_log_state(library_, HostLogger::get_instance().state(), &error) != 0)
            throw std::runtime_error(error ? error : "kernel runtime logger binding failed");
        auto create = symbol<decltype(&create_device_context)>(library_, "create_device_context");
        destroy_ = symbol<decltype(destroy_)>(library_, "destroy_device_context");
        finalize_ = symbol<decltype(finalize_)>(library_, "finalize_device");
        init_ = symbol<decltype(init_)>(library_, "simpler_kernel_mode_init");
        prepare_ = symbol<decltype(prepare_)>(library_, "simpler_kernel_mode_prepare_callable");
        launch_ = symbol<decltype(launch_)>(library_, "simpler_kernel_mode_launch");
        supported_ = symbol<decltype(supported_)>(library_, "simpler_kernel_mode_supported");
        committed_memory_ = symbol<decltype(committed_memory_)>(library_, "committed_device_memory_ctx");
        ctx_ = create();
        if (!ctx_) throw std::runtime_error("kernel runtime returned a null context");
    } catch (...) {
        dlclose(library_);
        library_ = nullptr;
        throw;
    }
}

KernelHostOwner::~KernelHostOwner() {
    if (library_)
        std::fputs("KernelHostOwner requires explicit close; context and runtime SO remain pinned.\n", stderr);
}

int KernelHostOwner::initialize(
    int device, const std::vector<uint8_t> &cpu, const std::vector<uint8_t> &core,
    const std::vector<uint8_t> &dispatcher, const CallConfig &config, uint64_t generation
) {
    std::lock_guard<std::mutex> lock(gate_);
    if (state_ != State::Created) return PTO_RUNTIME_ERR_INVALID_STATE;
    // Even a failed init may own native resources; only close is then legal.
    state_ = State::Closing;
    device_ = device;
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = init_(
            ctx_, device, cpu.data(), cpu.size(), core.data(), core.size(),
            dispatcher.empty() ? nullptr : dispatcher.data(), dispatcher.size(), &config, generation
        );
    } catch (...) {}
    if (rc == 0) state_ = State::Ready;
    return rc;
}

int KernelHostOwner::prepare(int32_t cid, const void *callable, size_t bytes) {
    std::lock_guard<std::mutex> lock(gate_);
    if (state_ != State::Ready || pending_ != 0) return PTO_RUNTIME_ERR_INVALID_STATE;
    try {
        return prepare_(ctx_, cid, callable, bytes);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int KernelHostOwner::launch(int32_t cid, const ChipStorageTaskArgs *args, void *stream) {
    std::lock_guard<std::mutex> lock(gate_);
    if (state_ != State::Ready || pending_ != 0) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (!args || !stream) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return launch_(ctx_, cid, args, stream);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int KernelHostOwner::close() {
    std::lock_guard<std::mutex> lock(gate_);
    if (state_ == State::Closed) return 0;
    if (pending_ != 0) return PTO_RUNTIME_ERR_INVALID_STATE;
    state_ = State::Closing;
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = finalize_(ctx_);
    } catch (...) {}
    if (rc != 0) return rc;
    destroy_(ctx_);
    ctx_ = nullptr;
    dlclose(library_);
    library_ = nullptr;
    state_ = State::Closed;
    return 0;
}

bool KernelHostOwner::ready() const {
    std::lock_guard<std::mutex> lock(gate_);
    return state_ == State::Ready;
}
bool KernelHostOwner::closed() const {
    std::lock_guard<std::mutex> lock(gate_);
    return state_ == State::Closed;
}
int KernelHostOwner::device() const {
    std::lock_guard<std::mutex> lock(gate_);
    return device_;
}
size_t KernelHostOwner::pending() const {
    std::lock_guard<std::mutex> lock(gate_);
    return pending_;
}
int KernelHostOwner::supported() const {
    std::lock_guard<std::mutex> lock(gate_);
    if (state_ != State::Ready) return 0;
    return supported_(ctx_);
}

size_t KernelHostOwner::committed_device_memory() const {
    std::lock_guard<std::mutex> lock(gate_);
    return ctx_ ? committed_memory_(ctx_) : 0;
}

std::shared_ptr<KernelQueuedInvocation> KernelHostOwner::accept(
    int32_t cid, const ChipTensor *tensors, size_t tensor_count, const uint64_t *scalars, size_t scalar_count
) {
    if (cid < 0 || cid >= MAX_REGISTERED_CALLABLE_IDS || tensor_count > CHIP_MAX_TENSOR_ARGS ||
        scalar_count > CHIP_MAX_SCALAR_ARGS ||
        !valid_invocation_counts(static_cast<int32_t>(tensor_count), static_cast<int32_t>(scalar_count)) ||
        (tensor_count && !tensors) || (scalar_count && !scalars))
        throw std::invalid_argument("invalid kernel invocation id/counts");
    auto call = std::shared_ptr<KernelQueuedInvocation>(
        new KernelQueuedInvocation(shared_from_this(), cid, tensors, tensor_count, scalars, scalar_count)
    );
    {
        std::lock_guard<std::mutex> lock(gate_);
        if (state_ != State::Ready || pending_ == std::numeric_limits<size_t>::max())
            throw std::runtime_error("kernel owner does not accept invocations");
        ++pending_;
        call->admitted_ = true;
    }
    return call;
}

KernelQueuedInvocation::KernelQueuedInvocation(
    std::shared_ptr<KernelHostOwner> owner, int32_t cid, const ChipTensor *tensors, size_t tensor_count,
    const uint64_t *scalars, size_t scalar_count
) :
    owner_(std::move(owner)),
    cid_(cid) {
    if (tensor_count) tensors_.assign(tensors, tensors + tensor_count);
    if (scalar_count) scalars_.assign(scalars, scalars + scalar_count);
}

KernelQueuedInvocation::~KernelQueuedInvocation() {
    std::lock_guard<std::mutex> lock(owner_->gate_);
    if (admitted_ && !done_) --owner_->pending_;
}

int KernelQueuedInvocation::invoke(void *stream, int (*before)(void *) noexcept, void *data) noexcept {
    try {
        std::lock_guard<std::mutex> lock(owner_->gate_);
        if (done_) return result_;
        result_ = PTO_RUNTIME_ERR_INTERNAL;
        try {
            if (stream && admitted_ && owner_->state_ == KernelHostOwner::State::Ready) {
                result_ = before ? before(data) : 0;
                if (result_ == 0) {
                    ChipStorageTaskArgs args{};
                    for (const auto &tensor : tensors_)
                        args.add_tensor(tensor);
                    for (auto scalar : scalars_)
                        args.add_scalar(scalar);
                    result_ = owner_->launch_(owner_->ctx_, cid_, &args, stream);
                }
            }
        } catch (...) {
            result_ = PTO_RUNTIME_ERR_INTERNAL;
        }
        done_ = true;
        if (admitted_) --owner_->pending_;
        return result_;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}
}  // namespace simpler::kernel
