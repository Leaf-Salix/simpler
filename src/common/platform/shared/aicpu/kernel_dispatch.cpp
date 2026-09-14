/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#include "kernel_dispatch_args.h"

#include <cstring>
#include <limits>

#include "aicpu/cache_maintenance.h"
#include "aicpu/kernel_invocation_consumer.h"
#include "callable_protocol.h"

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_kernel_exec(void *arg) {
    if (arg == nullptr || reinterpret_cast<uintptr_t>(arg) % alignof(SimplerKernelDispatchArgs) != 0) return 1;
    const auto &outer = *static_cast<const SimplerKernelDispatchArgs *>(arg);
    const auto &h = outer.invocation;
    if (outer.packet_bytes < sizeof(outer) || outer.packet_bytes > std::numeric_limits<size_t>::max() ||
        h.mode != SIMPLER_MODE_KERNEL || h.callable_id < 0 || h.callable_id >= MAX_REGISTERED_CALLABLE_IDS ||
        h.generation == 0 || h.tensor_count < 0 || h.scalar_count < 0 || h.host_copy_tensor_count != 0 ||
        outer.round_epoch == 0 || h.payload_bytes != outer.packet_bytes - sizeof(outer) ||
        outer.residency_address == 0 ||
        outer.residency_address % alignof(KernelCallableDeviceResidency) != 0 ||
        outer.residency_address > std::numeric_limits<uintptr_t>::max() - sizeof(KernelCallableDeviceResidency))
        return 1;

    auto *descriptor = reinterpret_cast<const KernelCallableDeviceResidency *>(outer.residency_address);
    cache_invalidate_range(descriptor, sizeof(*descriptor));
    KernelCallableDeviceResidency resident{};
    std::memcpy(&resident, descriptor, sizeof(resident));
    if (!kernel_callable_residency_matches(resident, h.callable_id, h.generation)) return 3;

    const auto *payload = reinterpret_cast<const uint8_t *>(arg) + sizeof(outer);
    return consume_kernel_invocation(
        h, resident, outer.residency_address, outer.round_epoch, payload, static_cast<size_t>(h.payload_bytes)
    );
}
