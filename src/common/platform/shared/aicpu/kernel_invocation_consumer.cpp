/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#include "aicpu/kernel_invocation_consumer.h"

// The HBG executor deliberately has no TMR packet consumer. The symbol is
// weak so a TMR runtime object can provide the strong implementation without
// changing the common loader or the program entry.
__attribute__((weak)) int consume_kernel_invocation(
    const SimplerKernelInvocationHeader &, const KernelCallableDeviceResidency &, uint64_t, uint64_t, const uint8_t *, size_t
) noexcept {
    return 4; // KernelDispatchStatus::UnsupportedPayload
}
