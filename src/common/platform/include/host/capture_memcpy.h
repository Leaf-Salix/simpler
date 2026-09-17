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

#include <acl/acl_rt.h>

// These context-owned uploads must complete before their host scratch buffers
// die. They are not graph nodes: their device allocations survive graph replay.
// Restore the caller's thread mode even when the synchronous copy fails.
inline int capture_memcpy_h2d(void *dst, size_t dst_bytes, const void *src, size_t src_bytes) {
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    const aclError exchange_rc = aclmdlRICaptureThreadExchangeMode(&mode);
    if (exchange_rc != ACL_SUCCESS) return static_cast<int>(exchange_rc);
    const aclError copy_rc = aclrtMemcpy(dst, dst_bytes, src, src_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    const aclError restore_rc = aclmdlRICaptureThreadExchangeMode(&mode);
    return static_cast<int>(restore_rc != ACL_SUCCESS ? restore_rc : copy_rc);
}
