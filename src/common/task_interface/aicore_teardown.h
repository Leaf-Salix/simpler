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

constexpr uint32_t AICORE_POST_CLOSE_RELEASE = 1;
constexpr uint32_t AICORE_PRE_WINDOW_HOST_CANCEL = UINT32_MAX;
constexpr uint32_t AICORE_PRE_WINDOW_CANCEL_POLL_INTERVAL = 256;
static_assert((AICORE_PRE_WINDOW_CANCEL_POLL_INTERVAL & (AICORE_PRE_WINDOW_CANCEL_POLL_INTERVAL - 1)) == 0);

// A pre-window cancellation uses UINT32_MAX; ordinary A2/A3 retirement uses 1
// after window-close. AICore only bypass-loads this isolated line. The host
// clears it before launch and may cancel only before AICPU submission; AICPU
// admission failures may cancel before opening any register window.
struct alignas(64) AicoreTeardownControl {
    uint32_t post_close_release;
};

static_assert(sizeof(AicoreTeardownControl) == 64);
static_assert(alignof(AicoreTeardownControl) == 64);
static_assert(std::is_standard_layout_v<AicoreTeardownControl>);
static_assert(std::is_trivially_copyable_v<AicoreTeardownControl>);
