#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""wide_fanin: stress-test the fanin dedup contains() scan at various K values.

Case 1 (WideFaninK15):   15 producers → consumer reads all 15 + Y → K=15
Case 2 (ExplicitDepK17): 16 producers → consumer explicit deps(16) + 1 tensormap → K=17
Case 3 (WideFaninK64):   64 producers → consumer reads all 64 + Y → K=64

Tensor layout: x_0..x_63, y  (65 tensors total).
  Y is always at slot 64 (the last tensor).
  COPY_FIRST_TO_LAST kernel: reads args[0], writes args[Y_IDX].
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

N_PRODUCERS_MAX = 127
SENTINEL = 42.0
INIT_VAL = -1.0


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestWideFanin(SceneTestCase):
    """wide_fanin: high-fanin dependency resolution under tensormap dedup."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/wide_fanin_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * (N_PRODUCERS_MAX + 1),  # 65 slots
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_CONST",
                "source": "../dummy_task/kernels/aic/kernel_write_const.cpp",
                "core_type": "aic",
            },
            {
                "func_id": 1,
                "name": "COPY_FIRST",
                "source": "../dummy_task/kernels/aic/kernel_copy_first.cpp",
                "core_type": "aic",
            },
            {
                "func_id": 2,
                "name": "COPY_FIRST_TO_LAST",
                "source": "kernels/aic/kernel_copy_first_to_last.cpp",
                "core_type": "aic",
            },
        ],
    }

    CASES = [
        {
            "name": "ExplicitDepK17",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 2, "n_producers": 16},
        },
        {
            "name": "WideFaninK64",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 3, "n_producers": 64},
        },
        {
            "name": "ExplicitDepK128",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 3, "n_producers": 127},
        },
    ]

    def generate_args(self, params):
        tensors = [
            Tensor(f"x_{i}", torch.full((16,), INIT_VAL, dtype=torch.float32))
            for i in range(N_PRODUCERS_MAX)
        ]
        tensors.append(Tensor("y", torch.full((16,), INIT_VAL, dtype=torch.float32)))
        return TaskArgsBuilder(
            *tensors,
            Scalar("case", int(params["case"])),
            Scalar("n_producers", int(params["n_producers"])),
        )

    def compute_golden(self, args, params):
        n = int(params["n_producers"])
        for i in range(n):
            getattr(args, f"x_{i}")[0] = SENTINEL
        args.y[0] = SENTINEL


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
