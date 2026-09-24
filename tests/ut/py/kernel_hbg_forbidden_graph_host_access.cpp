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

#include <cstdint>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

void read_device_tensor_in_graph_body(const GraphTaskArgs &args) {
    const uint32_t index[] = {0};
    const uint64_t step = get_tensor_data<uint64_t>(args.tensor(0).ref(), 1, index);

    // The queued Graph recorder must unwind rather than continue with a
    // fabricated value. Keeping the value in control flow makes a regression
    // fail by timeout instead of accidentally returning INVALID_ARGS.
    volatile uint64_t progress = 0;
    while (progress < 1)
        progress += step;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 1};
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    GraphTaskArgs graph_args;
    graph_args.add_input(args.tensor(0).ref());
    rt_submit_graph(&read_device_tensor_in_graph_body, graph_args);
}

}  // extern "C"
