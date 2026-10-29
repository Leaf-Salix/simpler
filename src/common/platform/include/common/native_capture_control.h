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

#include <cstdint>
#include <type_traits>

// This allocation tail is never restored by graph publication. A replay gate
// advances the epoch and preserves the first device failure across replays.
struct alignas(64) NativeCaptureControl {
    uint64_t execution_epoch{0};
    int32_t sticky_error{0};
    uint32_t remaining_threads{0};
    uint64_t padding[6]{};
};
static_assert(sizeof(NativeCaptureControl) == 64);

struct NativeCaptureGateArgs {
    uint64_t control;
    uint32_t threads;
    uint32_t reserved{0};
};

static_assert(std::is_trivially_copyable_v<NativeCaptureControl> && std::is_standard_layout_v<NativeCaptureControl>);
static_assert(sizeof(NativeCaptureGateArgs) == 16);
static_assert(std::is_trivially_copyable_v<NativeCaptureGateArgs> && std::is_standard_layout_v<NativeCaptureGateArgs>);
