#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""moe_aggregate: verify high-fanin aggregation under tensormap dedup.

MoE aggregation pattern: K producers each write a distinct expert tensor
(experts[i][0] = 42.0), then a single consumer sums all K expert outputs
into Y[0].  Tests K=1..16 fanin to stress the tensormap contains() linear
scan at increasing edge counts.

Case K=1 (CopyFirst):
  1 producer writes experts[0][0] = 42.0.
  Consumer copies experts[0][0] -> Y[0] via COPY_FIRST.
  expect Y[0] = 42.0

Case K=8 (Aggregate):
  8 producers write experts[0..7][0] = 42.0.
  Zero task writes Y[0] (barrier for INOUT ordering).
  Consumer aggregates: Y[0] = sum(experts[0..7][0]) = 336.0.
  expect Y[0] = 336.0

Case K=15 (Aggregate, MAX_TENSOR_ARGS limit):
  15 producers write experts[0..14][0] = 42.0.
  Consumer aggregates: Y[0] = sum(experts[0..14][0]) = 630.0.
  15 INPUT + 1 INOUT = 16 tensor args (at MAX_TENSOR_ARGS cap).
  expect Y[0] = 630.0

Case K=16 (Aggregate, explicit deps):
  16 producers write experts[0..15][0] = 42.0.
  Consumer uses set_dependencies(producer_ids, 16) + 1 tensormap edge = K=17.
  Consumer aggregates: Y[0] = sum(experts[0..15][0]) = 672.0.
  expect Y[0] = 672.0
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

MAX_TENSOR_ARGS = 16
SENTINEL = 42.0
INIT_VAL = -1.0  # distinguishable from sentinel in unmodified slots


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestMoEAggregate(SceneTestCase):
    """moe_aggregate: high-fanin aggregation stress test for contains()."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/moe_aggregate_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN] * MAX_TENSOR_ARGS + [D.INOUT] + [D.SCALAR],
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
                "name": "AGGREGATE",
                "source": "kernels/aic/kernel_aggregate.cpp",
                "core_type": "aic",
            },
        ],
    }

    CASES = [
        {
            "name": "K01_CopyFirst",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"K": 1},
        },
        {
            "name": "K08_Aggregate",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"K": 8},
        },
        {
            "name": "K15_Aggregate",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"K": 15},
        },
        {
            "name": "K16_ExplicitDep",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 2, "block_dim": 1},
            "params": {"K": 16},
        },
    ]

    def generate_args(self, params):
        """Build 16 expert tensors + Y + K scalar.

        All 16 expert tensor slots are populated even when K < 16 (unused
        slots carry INIT_VAL; the kernel reads only the first K).
        """
        tensors = [
            Tensor(f"experts_{i}", torch.full((16,), INIT_VAL, dtype=torch.float32))
            for i in range(MAX_TENSOR_ARGS)
        ]
        tensors.append(Tensor("y", torch.full((16,), INIT_VAL, dtype=torch.float32)))
        return TaskArgsBuilder(*tensors, Scalar("K", int(params["K"])))

    def compute_golden(self, args, params):
        """Producers write SENTINEL to each expert; consumer sums K experts into Y."""
        k = params["K"]
        for i in range(k):
            getattr(args, f"experts_{i}")[0] = SENTINEL
        args.y[0] = k * SENTINEL


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
