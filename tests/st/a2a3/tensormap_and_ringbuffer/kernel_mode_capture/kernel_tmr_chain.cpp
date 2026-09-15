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

#include "orchestration_api.h"

extern "C" void kernel_tmr_chain(const ChipTaskArgs &args) {
    SIMPLER_SCOPE_GUARD();
    uint32_t shape[] = {128 * 128};
    TensorCreateInfo intermediate(shape, 1, DataType::FLOAT32);
    auto previous = args.tensor(0).ref();
    for (int step = 0; step < 16; ++step) {
        CoreTaskArgs task;
        task.add_input(previous);
        if (step == 15) {
            task.add_output(args.tensor(1).ref());
        } else {
            task.add_output(intermediate);
        }
        task.add_scalar(args.scalar(0));
        auto outputs = rt_submit_aiv_task(0, task);
        if (step < 15) previous = outputs.get_ref(0);
    }
}
