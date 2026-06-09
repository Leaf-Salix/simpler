#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""wide_fanin: verify high-fanin dependency resolution under tensormap dedup.

Case 1 (WideFaninBarrier):
  15 producers each write to its own tensor X_i[0]=42.0 via WRITE_CONST.
  The consumer adds all 15 as inputs (add_input) plus Y as output (add_inout).
  COPY_FIRST copies X_0[0] -> Y[0].  This yields K=15 fanin candidates from
  tensormap lookup alone -- one edge per distinct tensor registered in the
  tensormap.  Tensor arg budget: 15 INPUT + 1 INOUT = 16 = MAX_TENSOR_ARGS.

Case 2 (ExplicitDepWideFanin):
  16 producers each write X_i[0]=42.0.  The consumer uses set_dependencies()
  on the primitive Arg API (not ArgWithDeps) to attach all 16 producer IDs as
  explicit deps, PLUS add_input(X_0) for one tensormap edge.  Total fanin =
  16 explicit + 1 tensormap = 17.  COPY_FIRST copies X_0[0] -> Y[0].

Golden:
  Y[0] = 42.0 (copied from X_0[0] which was written by producer_0).
  X_i[0] = 42.0 for i in 0..N-1 (each written by its own producer).
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

MAX_TENSOR_ARGS = 16
SENTINEL = 42.0
INIT_VAL = -1.0  # distinguishable from sentinel in unmodified slots


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestWideFanin(SceneTestCase):
    """wide_fanin: high-fanin dependency resolution under tensormap dedup."""

    RTOL = 0
    ATOL = 0

    # case=1: N=15 producers, 1 output tensor + 1 scalar = 17 args
    # case=2: N=16 producers, 1 output tensor + 1 scalar = 18 args
    # Signature uses max(15,16)+1 = 17 tensor directions; extra slot unused
    # by case=1 (which only accesses tensors 0..15 inclusive).
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/wide_fanin_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT] * (MAX_TENSOR_ARGS + 1),  # 17 slots: X_0..X_15, Y
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
            "name": "WideFaninBarrier",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 1},
        },
        {
            "name": "ExplicitDepWideFanin",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"case": 2},
        },
    ]

    def generate_args(self, params):
        """Build producer tensors + output tensor + scalar.

        Both cases use 17 tensor slots (x_0..x_15, y) + 1 scalar = 18 args.
        case=1 only uses x_0..x_14 + y (16 tensors); x_15 is unused but present.
        """
        tensors = [
            Tensor(f"x_{i}", torch.full((16,), INIT_VAL, dtype=torch.float32))
            for i in range(16)
        ]
        tensors.append(Tensor("y", torch.full((16,), INIT_VAL, dtype=torch.float32)))
        return TaskArgsBuilder(
            *tensors,
            Scalar("case", int(params["case"])),
        )

    def compute_golden(self, args, params):
        """Every producer writes SENTINEL to its tensor[0]; consumer copies X_0[0] -> Y[0]."""
        n_producers = 15 if params["case"] == 1 else 16
        for i in range(n_producers):
            getattr(args, f"x_{i}")[0] = SENTINEL
        args.y[0] = SENTINEL


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
