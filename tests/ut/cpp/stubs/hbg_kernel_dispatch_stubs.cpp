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
#include <cstdlib>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "host_build_graph/runtime.h"

// Failure-path tests run the production consumer and sim participant gate, but
// must never reach an executor or perform hardware initialization.
void prepare_kernel_aicpu_thread() {}

int32_t aicpu_execute_restored_graph(Runtime *) { std::abort(); }

static uint64_t dispatch_test_regs = 0;

void set_platform_regs(uint64_t regs) { dispatch_test_regs = regs; }

uint64_t get_platform_regs() { return dispatch_test_regs; }
