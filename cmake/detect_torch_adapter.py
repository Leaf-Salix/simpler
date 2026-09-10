# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inspect installed build dependencies without importing torch_npu."""

import importlib.metadata
import importlib.util
import sys
from pathlib import Path


def main():
    if importlib.util.find_spec("torch") is None:
        return
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        return
    torch = importlib.import_module("torch")
    if not torch.__file__:
        return

    root = Path(torch.__file__).resolve().parent
    npu = Path(spec.origin).resolve().parent
    required = [
        root / "include/torch/csrc/autograd/python_variable.h",
        npu / "include/torch_npu/csrc/framework/OpCommand.h",
        npu / "include/torch_npu/csrc/core/npu/NPUFormat.h",
        npu / "include/torch_npu/csrc/core/npu/NPUCachingAllocator.h",
        npu / "include/torch_npu/csrc/core/npu/NPUStream.h",
        npu / "lib/libtorch_npu.so",
        *[root / "lib" / name for name in ("libtorch.so", "libtorch_cpu.so", "libtorch_python.so", "libc10.so")],
    ]
    if not all(path.is_file() for path in required):
        print("Skipping optional Torch adapter: installed SDK headers/libraries are incomplete", file=sys.stderr)
        return
    print(root)
    print(npu)
    print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    print(torch.__version__)
    print(importlib.metadata.version("torch-npu"))


if __name__ == "__main__":
    main()
