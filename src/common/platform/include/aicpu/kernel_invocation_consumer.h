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

#include "kernel_dispatch_args.h"
#include "callable.h"

// Per-thread kernel-entry setup supplied by platforms with a scheduling policy.
void prepare_kernel_aicpu_thread();

// The binder pairs the ID with a committed device image. Before rebuilding the
// function table, the runtime consumer must establish cache visibility,
// validate child offsets against callable_bytes, and check signature counts.
// Child func_ids and resolved_addr fields provide the per-invocation mapping.
// This function owns neither the launch packet nor the committed image.
int consume_kernel_invocation(
    const SimplerKernelDispatchArgs &args, const ChipCallable &callable, size_t callable_bytes, const void *payload,
    size_t payload_bytes
);
