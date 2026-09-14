/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernel_invocation_header.h"

inline constexpr uint64_t SIMPLER_KERNEL_RESIDENCY_MAGIC = 0x534b524553494445ULL; // "SKRESIDE"

// Stable context binding produced by the K3 resource owner. It contains only
// device identities and capacities; callable identity/generation is layered
// on top by KernelCallableDeviceResidency. The view is host-readable while
// being assembled and can be copied verbatim into a device descriptor.
struct KernelStableDeviceBinding {
    uint64_t context_generation{0};
    uint64_t runtime_address{0};
    uint64_t sm_address{0};
    uint64_t sm_capacity{0};
    uint64_t arena_address{0};
    uint64_t arena_capacity{0};
    uint64_t runtime_offset{0};
    uint32_t aicpu_thread_count{0};
    uint32_t reserved{0};
};

inline bool kernel_stable_device_binding_valid(const KernelStableDeviceBinding &binding) noexcept {
    const bool sm_range = binding.sm_address <= std::numeric_limits<uint64_t>::max() - binding.sm_capacity;
    const bool arena_range = binding.arena_address <= std::numeric_limits<uint64_t>::max() - binding.arena_capacity;
    return binding.context_generation != 0 && binding.runtime_address != 0 && binding.sm_address != 0 &&
           binding.sm_capacity != 0 && binding.arena_address != 0 && binding.arena_capacity != 0 &&
           sm_range && arena_range && (binding.sm_address % alignof(uint64_t)) == 0 &&
           (binding.arena_address % alignof(uint64_t)) == 0 &&
           binding.runtime_offset < binding.arena_capacity && binding.aicpu_thread_count > 1 &&
           binding.reserved == 0;
}

// Device-readable, position-independent TMR binding published by the host
// during prepare. Addresses are opaque HBM VAs; no host pointer crosses this
// wire. A descriptor is immutable while a launch may reference it.
struct KernelCallableDeviceResidency {
    uint64_t magic{SIMPLER_KERNEL_RESIDENCY_MAGIC};
    uint64_t generation{0};
    uint64_t device_address{0};
    uint64_t bytes{0};
    int32_t callable_id{-1};
    uint32_t reserved{0};
    uint64_t runtime_address{0};
    uint64_t sm_address{0};
    uint64_t sm_capacity{0};
    uint64_t arena_address{0};
    uint64_t arena_capacity{0};
    uint64_t runtime_offset{0};
    uint64_t function_table_address{0};
    uint32_t function_table_count{0};
    uint32_t aicpu_thread_count{0};
    uint64_t context_generation{0};
};

// Host-to-AICPU registration payload. The descriptor itself remains in HBM;
// this small POD only publishes its address and immutable identity to the
// runtime-local device registry before any invocation can be admitted.
struct KernelCallableResidencyRegistration {
    uint64_t descriptor_address{0};
    uint64_t generation{0};
    uint64_t context_generation{0};
    int32_t callable_id{-1};
    uint32_t reserved{0};
};

static_assert(std::is_trivially_copyable_v<KernelCallableResidencyRegistration>);
static_assert(std::is_standard_layout_v<KernelCallableResidencyRegistration>);
static_assert(sizeof(KernelCallableResidencyRegistration) == 32);

static_assert(std::is_trivially_copyable_v<KernelCallableDeviceResidency>);
static_assert(std::is_standard_layout_v<KernelCallableDeviceResidency>);

inline bool kernel_callable_residency_matches(
    const KernelCallableDeviceResidency &resident, int32_t callable_id, uint64_t generation
) noexcept {
    const bool callable_range = resident.device_address <= std::numeric_limits<uint64_t>::max() - resident.bytes;
    const bool table_range = resident.function_table_address <=
                                 std::numeric_limits<uint64_t>::max() -
                                     static_cast<uint64_t>(resident.function_table_count) * sizeof(uint64_t);
    return resident.magic == SIMPLER_KERNEL_RESIDENCY_MAGIC && resident.reserved == 0 &&
           resident.callable_id == callable_id && resident.generation == generation && resident.generation != 0 &&
           resident.context_generation != 0 && resident.device_address != 0 && resident.bytes != 0 &&
           callable_range && table_range && (resident.device_address % alignof(uint64_t)) == 0 &&
           (resident.function_table_address % alignof(uint64_t)) == 0 &&
           resident.runtime_address != 0 && resident.sm_address != 0 && resident.sm_capacity != 0 &&
           resident.arena_address != 0 && resident.arena_capacity != 0 &&
           resident.function_table_address != 0 && resident.function_table_count != 0 && resident.aicpu_thread_count > 1;
}

inline bool kernel_callable_residency_matches(
    const SimplerKernelInvocationHeader &header, const KernelCallableDeviceResidency &resident
) noexcept {
    return kernel_callable_residency_matches(resident, header.callable_id, header.generation);
}
