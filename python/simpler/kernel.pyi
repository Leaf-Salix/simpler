# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from collections.abc import Iterable, Sequence

from _task_interface import CallConfig

from simpler.task_interface import ChipWorker

class KernelHostOwner:
    def initialize(
        self,
        device: int,
        aicpu_binary: bytes,
        aicore_binary: bytes,
        dispatcher_binary: bytes,
        config: CallConfig,
        context_generation: int,
    ) -> int: ...
    def prepare(self, callable_id: int, callable: bytes) -> int: ...
    def close(self) -> int: ...
    def supported(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    @property
    def pending(self) -> int: ...

def create_owner(runtime_path: str) -> KernelHostOwner: ...
def enqueue(
    owner: KernelHostOwner | ChipWorker,
    callable_id: int,
    tensors: Iterable[object],
    scalar_bits: Sequence[int] = (),
    *,
    op_name: str = "simpler_kernel",
) -> None: ...
