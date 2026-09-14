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

#include "task_interface/kernel_callable_residency.h"
#include "task_interface/tmr_kernel_context.h"
#include "host/kernel_device_resources.h"
#include "tensormap_and_ringbuffer/kernel_clear_plan.h"

// Kernel-only owner for the stable execution views consumed by K4/K5/K7.
// It deliberately owns no launch operation and exposes no mutable device
// resource. DeviceRunnerBase remains responsible for allocation; this object
// defines the publication/borrow/close lifetime boundary.
class KernelContextProvider {
public:
    KernelContextProvider() = default;
    KernelContextProvider(const KernelContextProvider &) = delete;
    KernelContextProvider &operator=(const KernelContextProvider &) = delete;

    int prepare(const KernelResourceLayout &layout, const KernelResourceOps &ops) noexcept {
        if (resources_.prepared() || resources_.has_live_resources()) return -1;
        return resources_.prepare(layout, ops);
    }

    int freeze() noexcept { return resources_.freeze(); }

    // Kernel Runtime currently owns the physical SM/arena allocations because
    // program mode still uses the same Runtime owner.  This path records that
    // already-validated binding without allocating a second copy, while still
    // forcing kernel callers through the provider's prepare -> freeze ->
    // publish lifecycle.  The provider owns the published view, not the
    // borrowed Runtime allocation.
    int prepare_borrowed_binding(const KernelStableDeviceBinding &binding) noexcept {
        if (published_ || borrowed_binding_prepared_ || !kernel_stable_device_binding_valid(binding)) return -1;
        binding_ = binding;
        borrowed_binding_prepared_ = true;
        return 0;
    }

    int freeze_borrowed_binding() noexcept {
        if (!borrowed_binding_prepared_ || published_) return -1;
        published_ = true;
        return 0;
    }

    int borrow(uint64_t schema, const uint64_t *required, size_t count, KernelResourceBinding &out) const noexcept {
        if (!published_ || !resources_.frozen()) return -1;
        return resources_.bind(schema, required, count, out);
    }

    int publish_binding(const KernelStableDeviceBinding &binding) noexcept {
        if (published_ || borrowed_binding_prepared_ || !resources_.frozen() ||
            !kernel_stable_device_binding_valid(binding))
            return -1;
        binding_ = binding;
        published_ = true;
        return 0;
    }

    int publish_clear_plan(const simpler::tmr::TmrKernelClearPlan &plan) noexcept {
        if (!published_ || clear_published_ || plan.context_generation != binding_.context_generation) return -1;
        clear_plan_ = plan;
        clear_published_ = true;
        return 0;
    }

    int publish_context_descriptor(const simpler::tmr::TmrKernelContextDescriptor &descriptor) noexcept {
        if (!published_ || descriptor_published_ || descriptor.context_generation != binding_.context_generation ||
            descriptor.self_address == 0 || descriptor.resident_runtime == 0 || descriptor.resident_kernel_args == 0 ||
            descriptor.sm_base != binding_.sm_address || descriptor.sm_capacity != binding_.sm_capacity ||
            descriptor.arena_base != binding_.arena_address || descriptor.arena_capacity != binding_.arena_capacity ||
            descriptor.runtime_offset != binding_.runtime_offset || descriptor.control_address == 0 ||
            descriptor.reports_address == 0 || descriptor.control_bytes == 0 || descriptor.reports_bytes == 0)
            return -1;
        descriptor_ = descriptor;
        descriptor_published_ = true;
        return 0;
    }

    int adopt_tmr_storage(KernelResourceOps ops, void *descriptor, void *table, void *control, void *reports) noexcept {
        if (!published_ || storage_owned_ || !ops.valid() || descriptor == nullptr || table == nullptr ||
            control == nullptr || reports == nullptr)
            return -1;
        storage_ops_ = ops;
        storage_ = {descriptor, table, control, reports};
        storage_owned_ = true;
        return 0;
    }

    const KernelStableDeviceBinding *binding_view() const noexcept { return published_ ? &binding_ : nullptr; }

    const simpler::tmr::TmrKernelClearPlan *clear_plan_view() const noexcept {
        return clear_published_ ? &clear_plan_ : nullptr;
    }

    const simpler::tmr::TmrKernelContextDescriptor *descriptor_view() const noexcept {
        return descriptor_published_ ? &descriptor_ : nullptr;
    }

    bool published() const noexcept { return published_; }
    bool owns_tmr_storage() const noexcept { return storage_owned_; }
    bool close() noexcept {
        if (resources_.close() != 0) return false;
        if (storage_owned_) {
            int rc = 0;
            for (void **ptr : {&storage_.descriptor, &storage_.table, &storage_.control, &storage_.reports}) {
                if (*ptr == nullptr) continue;
                int release_rc = -1;
                try {
                    release_rc = storage_ops_.release(storage_ops_.context, *ptr);
                } catch (...) {
                    // A release callback is supplied by the platform allocator.
                    // Keep the pointer owned when a test or adapter violates
                    // the non-throwing callback contract so close() remains a
                    // retryable, fail-closed operation rather than terminating
                    // through this noexcept boundary.
                }
                if (release_rc != 0) rc = release_rc;
                else *ptr = nullptr;
            }
            if (rc != 0) return false;
            storage_owned_ = false;
            storage_ops_ = {};
        }
        published_ = false;
        borrowed_binding_prepared_ = false;
        clear_published_ = false;
        descriptor_published_ = false;
        binding_ = {};
        clear_plan_ = {};
        descriptor_ = {};
        return true;
    }

private:
    struct TmrStorage {
        void *descriptor{nullptr};
        void *table{nullptr};
        void *control{nullptr};
        void *reports{nullptr};
    };
    KernelDeviceResources resources_{};
    KernelResourceOps storage_ops_{};
    TmrStorage storage_{};
    bool storage_owned_{false};
    KernelStableDeviceBinding binding_{};
    simpler::tmr::TmrKernelClearPlan clear_plan_{};
    simpler::tmr::TmrKernelContextDescriptor descriptor_{};
    bool published_{false};
    bool borrowed_binding_prepared_{false};
    bool clear_published_{false};
    bool descriptor_published_{false};
};
