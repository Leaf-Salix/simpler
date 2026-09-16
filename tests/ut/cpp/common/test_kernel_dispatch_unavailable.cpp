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
#include <gtest/gtest.h>
#include "kernel_dispatch_args.h"
#include "callable.h"
#include "callable_protocol.h"

TEST(KernelDispatchUnavailable, ProductionConsumerDoesNotReportExecutionSuccess) {
    ChipCallable callable{};
    SimplerKernelDispatchArgs packet{};
    packet.chip_callable_address = reinterpret_cast<uint64_t>(&callable);
    packet.chip_callable_bytes = sizeof(callable);
    packet.packet_bytes = sizeof(packet);
    packet.invocation.mode = SIMPLER_MODE_KERNEL;
    packet.invocation.callable_id = MAX_REGISTERED_CALLABLE_IDS - 1;
    EXPECT_EQ(simpler_aicpu_kernel_exec(&packet), 2);
    packet.invocation.callable_id = MAX_REGISTERED_CALLABLE_IDS;
    EXPECT_EQ(simpler_aicpu_kernel_exec(&packet), 2);
}
