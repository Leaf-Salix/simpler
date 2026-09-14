/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "kernel_callable_residency.h"
#include "kernel_invocation_header.h"

// CANN copies this complete POD prefix plus the packet payload into the
// AICPU launch argument area. The descriptor address is checked against the
// current context's registered slot before it is dereferenced.
struct SimplerKernelDispatchArgs {
    uint64_t packet_bytes{0};
    uint64_t residency_address{0};
    // Host-generated invocation epoch. It is deliberately outside K9: the
    // epoch distinguishes two replays of the same callable/generation and is
    // consumed only by the device round gate.
    uint64_t round_epoch{0};
    SimplerKernelInvocationHeader invocation{};
};

enum class KernelDispatchStatus : int32_t {
    Success = 0,
    InvalidArgs = 1,
    NotResident = 2,
    Stale = 3,
    UnsupportedPayload = 4,
    InvalidBinding = 5,
    ExecutionFailed = 6,
    CleanupFailed = 7,
    Busy = 8,
};

static_assert(std::is_trivially_copyable_v<SimplerKernelDispatchArgs>);
static_assert(std::is_standard_layout_v<SimplerKernelDispatchArgs>);

extern "C" int simpler_aicpu_kernel_exec(void *args);
