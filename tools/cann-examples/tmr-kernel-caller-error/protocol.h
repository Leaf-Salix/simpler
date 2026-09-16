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
struct alignas(64) CallerProbeReport {
    uint64_t work_complete;
    int32_t work_status;
    uint32_t reserved;
    uint8_t padding[48];
    uint64_t check_complete;
    int32_t observed_status;
    uint32_t check_reserved;
    uint8_t check_padding[48];
};
struct CallerProbeArgs {
    uint64_t report;
    int32_t status;
    uint32_t reserved;
};
static_assert(sizeof(CallerProbeReport) == 128);
static_assert(sizeof(CallerProbeArgs) == 16);
