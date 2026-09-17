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

using aclError = int32_t;
constexpr aclError ACL_SUCCESS = 0;
enum aclmdlRICaptureMode {
    ACL_MODEL_RI_CAPTURE_MODE_GLOBAL = 0,
    ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL,
    ACL_MODEL_RI_CAPTURE_MODE_RELAXED,
};
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE = 1 };

extern "C" aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode);
extern "C" aclError aclrtMemcpy(void *dst, size_t dst_bytes, const void *src, size_t src_bytes, aclrtMemcpyKind kind);
