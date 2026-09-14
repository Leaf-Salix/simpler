# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sensitivity checks for the numerical oracle, not evidence of NPU execution."""

import pytest

from tests.ut.py.kernel_capture_values import batch_expected

_SEQUENCE = (0, 1) * 50


def test_feedback_batch_expected_values():
    assert batch_expected(_SEQUENCE) == (200.0, 197.25, 50, 50)


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "swap"])
def test_feedback_batch_detects_single_execution_faults(mutation):
    expected = batch_expected(_SEQUENCE)
    for index in range(len(_SEQUENCE) - (mutation == "swap")):
        modified = list(_SEQUENCE)
        if mutation == "drop":
            modified.pop(index)
        elif mutation == "duplicate":
            modified.insert(index, modified[index])
        else:
            modified[index], modified[index + 1] = modified[index + 1], modified[index]
        assert batch_expected(modified) != expected, (mutation, index)
