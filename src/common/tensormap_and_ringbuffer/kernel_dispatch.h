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

#include <atomic>
#include <cstring>
#include <limits>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "callable.h"
#include "common/kernel_args.h"
#include "kernel_dispatch_args.h"
#include "kernel_callable_residency.h"
#include "kernel_execution_inputs.h"

namespace simpler::tmr {

static_assert(std::extent_v<decltype(ChipCallable::child_func_ids_)> == RUNTIME_MAX_FUNC_ID);

inline bool readable_kernel_region(uint64_t address, uint64_t bytes, size_t alignment) {
    return address != 0 && address % alignment == 0 && bytes != 0 &&
           bytes <= std::numeric_limits<uintptr_t>::max() - address;
}

inline void cancel_uninitialized_kernel_cores(Runtime &runtime) {
    auto *gates = runtime.get_teardown_gates();
    for (int32_t core = 0; core < runtime.get_worker_count(); ++core) {
        __atomic_store_n(&gates[core].post_close_release, AICORE_PRE_WINDOW_HOST_CANCEL, __ATOMIC_RELEASE);
        cache_flush_range(&gates[core], sizeof(gates[core]));
    }
}

// Each barrier includes every admitted affinity slot. The final barrier
// releases all EntryArgs/TensorRef borrowers before a following launch.
class KernelDispatchGroup {
public:
    void barrier(int32_t count) {
        const uint32_t epoch = epoch_.load(std::memory_order_acquire);
        if (arrivals_.fetch_add(1, std::memory_order_acq_rel) + 1 == count) {
            arrivals_.store(0, std::memory_order_relaxed);
            epoch_.fetch_add(1, std::memory_order_release);
        } else {
            while (epoch_.load(std::memory_order_acquire) == epoch) {}
        }
    }

    void reset_status(int status) { status_.store(status, std::memory_order_relaxed); }
    void record_failure(int status) {
        if (status == 0) return;
        int expected = 0;
        status_.compare_exchange_strong(expected, status, std::memory_order_acq_rel);
    }
    int status() const { return status_.load(std::memory_order_acquire); }

    uint64_t functions[RUNTIME_MAX_FUNC_ID]{};

private:
    std::atomic<int32_t> arrivals_{0};
    std::atomic<uint32_t> epoch_{0};
    std::atomic<int> status_{0};
};

inline InvocationStatus admit_kernel_dispatch(
    const SimplerKernelDispatchArgs &args, const KernelCallableDeviceResidency &residency, Runtime *runtime,
    KernelDispatchGroup &group
) {
    if (!readable_kernel_region(residency.device_address, residency.bytes, alignof(ChipCallable)) ||
        residency.bytes < sizeof(ChipCallable))
        return InvocationStatus::InvalidBinding;

    const auto *callable = reinterpret_cast<const ChipCallable *>(residency.device_address);
    cache_invalidate_range(callable, sizeof(*callable));
    int32_t tensors = 0;
    int32_t scalars = 0;
    auto status = kernel::derive_invocation_counts(callable->signature_, callable->sig_count_, &tensors, &scalars);
    if (status != InvocationStatus::Ok) return status;
    const uint64_t storage_bytes = residency.bytes - sizeof(ChipCallable);
    if (callable->binary_size_ > storage_bytes || callable->child_count_ < 0 ||
        callable->child_count_ > RUNTIME_MAX_FUNC_ID)
        return InvocationStatus::InvalidBinding;

    std::memset(group.functions, 0, sizeof(group.functions));
    for (int32_t index = 0; index < callable->child_count_; ++index) {
        const int32_t id = callable->child_func_ids_[index];
        const uint64_t offset = callable->child_offsets_[index];
        if (id < 0 || id >= RUNTIME_MAX_FUNC_ID || group.functions[id] != 0 || offset < callable->binary_size_ ||
            offset % CALLABLE_ALIGN != 0 || offset > storage_bytes || sizeof(CoreCallable) > storage_bytes - offset)
            return InvocationStatus::InvalidBinding;
        const auto *child = reinterpret_cast<const CoreCallable *>(callable->storage_ + offset);
        cache_invalidate_range(child, sizeof(*child));
        if (child->sig_count_ < 0 || child->sig_count_ > CORE_MAX_TENSOR_ARGS || child->resolved_addr_ == 0 ||
            CoreCallable::binary_data_offset() > storage_bytes - offset ||
            child->binary_size_ > storage_bytes - offset - CoreCallable::binary_data_offset())
            return InvocationStatus::InvalidBinding;
        group.functions[id] = reinterpret_cast<uint64_t>(child);
    }

    const KernelBindingView binding{
        {args.binding_address, args.context_generation},
        runtime,
        {runtime->get_gm_sm_ptr(), static_cast<size_t>(args.sm_bytes), static_cast<size_t>(args.sm_bytes)},
        {runtime->get_prebuilt_arena_base(), static_cast<size_t>(args.arena_bytes),
         static_cast<size_t>(args.arena_bytes)},
        runtime->get_prebuilt_runtime_offset()
    };
    const KernelCallableView prepared{
        {residency.callable_id, tensors, scalars, residency.generation}, {group.functions, RUNTIME_MAX_FUNC_ID}
    };
    static_assert(offsetof(SimplerKernelDispatchArgs, invocation) + sizeof(args.invocation) == sizeof(args));
    return admit_kernel_execution(
        {reinterpret_cast<const uint8_t *>(&args.invocation),
         sizeof(args.invocation) + static_cast<size_t>(args.invocation.payload_bytes)},
        prepared, binding
    );
}

inline int execute_tmr_kernel_dispatch(
    const SimplerKernelDispatchArgs &args, const KernelCallableDeviceResidency &residency, const void *payload,
    size_t payload_bytes, void (*configure_platform)(const KernelArgs &)
) {
    if (!readable_kernel_region(args.binding_address, sizeof(KernelArgs), alignof(KernelArgs)))
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);

    const auto *binding = reinterpret_cast<const KernelArgs *>(args.binding_address);
    cache_invalidate_range(binding, sizeof(*binding));
    KernelArgs kernel_args;
    std::memcpy(&kernel_args, binding, sizeof(kernel_args));
    Runtime *runtime = kernel_args.runtime_args;
    if (!readable_kernel_region(reinterpret_cast<uint64_t>(runtime), sizeof(runtime->dev), alignof(Runtime)))
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);
    cache_invalidate_range(runtime, sizeof(runtime->dev));
    if (runtime->get_worker_count() <= 0 || runtime->get_worker_count() > RUNTIME_MAX_WORKER)
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);

    const int32_t count = runtime->get_aicpu_allowed_cpu_count();
    const int32_t launched = runtime->get_aicpu_launch_count();
    if (payload != reinterpret_cast<const uint8_t *>(&args) + sizeof(args) ||
        payload_bytes != args.invocation.payload_bytes || args.context_generation == 0 || kernel_args.regs == 0 ||
        args.sm_bytes < sizeof(SharedMemoryHeader) || args.arena_bytes < sizeof(RuntimeContext) ||
        args.sm_bytes > std::numeric_limits<size_t>::max() || args.arena_bytes > std::numeric_limits<size_t>::max() ||
        count <= 0 || count > MAX_GATE_THREADS || count > PLATFORM_MAX_AICPU_THREADS ||
        count != runtime->get_aicpu_thread_num() || launched < count || launched > MAX_GATE_THREADS) {
        cancel_uninitialized_kernel_cores(*runtime);
        return static_cast<int>(KernelDispatchStatus::InvalidArgs);
    }
    if (!platform_aicpu_affinity_gate_filter(runtime->get_aicpu_allowed_cpus(), count, launched)) return 0;
    const bool leader = platform_aicpu_affinity_thread_idx() == 0;

    // One resident executor and one group serve the device runtime SO.
    // Its host owner serializes launches, including launches of distinct callables.
    static KernelDispatchGroup group;
    if (leader) {
        configure_platform(kernel_args);
        const auto status = admit_kernel_dispatch(args, residency, runtime, group);
        group.reset_status(status == InvocationStatus::Ok ? 0 : static_cast<int>(KernelDispatchStatus::InvalidArgs));
        if (status != InvocationStatus::Ok) cancel_uninitialized_kernel_cores(*runtime);
    }
    group.barrier(count);
    AicpuPhaseScope run_wall(AicpuPhase::RunWall);

    int thread_status = 0;
    if (group.status() == 0) {
        AicpuPhaseScope preamble(AicpuPhase::Preamble);
        thread_status = init_kernel_execution();
    }
    group.barrier(count);
    group.record_failure(thread_status);
    group.barrier(count);

    if (group.status() == 0) {
        AicpuPhaseScope graph_build(AicpuPhase::GraphBuild);
        thread_status = run_kernel_execution();
    }
    group.barrier(count);
    group.record_failure(thread_status);
    group.barrier(count);
    if (leader) {
        if (group.status() == 0) group.record_failure(kernel_execution_status());
        AicpuPhaseScope post_orch(AicpuPhase::PostOrch);
        release_kernel_execution();
    }
    group.barrier(count);
    return group.status();
}

}  // namespace simpler::tmr
