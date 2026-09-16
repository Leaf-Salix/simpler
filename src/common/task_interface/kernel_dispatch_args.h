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
#include <type_traits>

#include "kernel_invocation_header.h"

// CANN deep-copies this prefix and the following payload into each launch.
// This is a kernel invocation envelope, not the program entry's KernelArgs
// layout. The binder supplies every address here from the issuing context's
// own committed state; none of them comes from a caller-supplied image.
struct SimplerKernelDispatchArgs {
    uint64_t packet_bytes;
    // Filled from the issuing context's committed residency, not a Host image.
    // The immutable device image outlives every referencing launch and graph.
    uint64_t chip_callable_address;
    uint64_t chip_callable_bytes;
    /* The issuing context's device KernelArgs. It is the one pointer a
       kernel-mode entry gets, and everything the program-mode entry receives in
       its own launch argument hangs off it: the resident runtime, the per-core
       register table, and the profiling bases. Stable for the context's life. */
    uint64_t binding_address;
    uint64_t context_generation;
    /* Extents of the two context-static regions the resident runtime names.
       The runtime records their bases but not their sizes, and the device may
       not read a region to learn how far it may read. */
    uint64_t sm_bytes;
    uint64_t arena_bytes;
    SimplerKernelInvocationHeader invocation;
};

// Logical runtime classifications, distinct from the native CANN 0/2 domain.
// Values 2 and 3 are retired residency-descriptor classifications.
enum class KernelDispatchStatus : int32_t {
    Success = 0,
    InvalidArgs = 1,
    UnsupportedPayload = 4,
    InvalidBinding = 5,
    ExecutionFailed = 6,
    CleanupFailed = 7,
};

static_assert(
    std::is_trivially_copyable_v<SimplerKernelDispatchArgs> && std::is_standard_layout_v<SimplerKernelDispatchArgs>
);
static_assert(sizeof(SimplerKernelDispatchArgs) == 88);
static_assert(offsetof(SimplerKernelDispatchArgs, chip_callable_address) == 8);
static_assert(offsetof(SimplerKernelDispatchArgs, chip_callable_bytes) == 16);
static_assert(offsetof(SimplerKernelDispatchArgs, binding_address) == 24);
static_assert(offsetof(SimplerKernelDispatchArgs, context_generation) == 32);
static_assert(offsetof(SimplerKernelDispatchArgs, sm_bytes) == 40);
static_assert(offsetof(SimplerKernelDispatchArgs, arena_bytes) == 48);
static_assert(offsetof(SimplerKernelDispatchArgs, invocation) == 56);

extern "C" int simpler_aicpu_kernel_exec(void *args);
