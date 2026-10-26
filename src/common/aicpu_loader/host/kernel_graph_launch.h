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

#include "host_build_graph/kernel_graph_prepared.h"

namespace hbg {

inline int launch_graph_prepared(
    const GraphPreparedDispatch &source, aclrtFuncHandle function, uint32_t blocks, aclrtStream stream
) {
    GraphPreparedDispatch args = source;
    return aclrtLaunchKernelWithHostArgs(function, blocks, stream, nullptr, &args, sizeof(args), nullptr, 0);
}

}  // namespace hbg
