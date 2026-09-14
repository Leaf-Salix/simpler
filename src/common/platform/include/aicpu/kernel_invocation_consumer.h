/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#pragma once

#include <cstddef>

#include "kernel_callable_residency.h"
#include "kernel_invocation_header.h"

// Runtime-specific implementation. host_build_graph supplies the fail-closed
// stub; tensormap_and_ringbuffer supplies the TMR executor implementation.
// `residency_address` is the device address from the dispatch envelope. The
// consumer must match it against the binding reference in the packet before
// constructing any executor view; the copied descriptor alone is not an
// authority for an arbitrary packet address.
int consume_kernel_invocation(
    const SimplerKernelInvocationHeader &invocation, const KernelCallableDeviceResidency &resident,
    uint64_t residency_address, uint64_t round_epoch, const uint8_t *payload, size_t payload_bytes
) noexcept;
