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

#include "worker/runtime_c_api.h"

// Internal host-runtime hook, not a dlsym lifecycle API. Input is borrowed and
// immutable during the call; output is caller-exclusive and unchanged on error.
// No device resources are acquired or retained.
//
// A context's kernel contract is a pure function of its context-static config,
// so the init entry calls this once per context and the answer holds for that
// context's lifetime. The implementations keep no static or thread-local state,
// which is what makes that single call free of ordering constraints against any
// other context's — it is not an invitation to recompute a context's own
// contract on a later path.
//
// TMR rejects invalid config with INVALID_ARGUMENT and a null output or invalid
// generated contract with INTERNAL. Unsupported implementations return UNSUPPORTED.
extern "C" int build_kernel_pipeline_contract_impl(const CallConfig *config, PipelineContract *out);
