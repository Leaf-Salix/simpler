# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)
project(kernel_adapter_probe CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir OUTPUT_VARIABLE nb_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED PATHS ${nb_dir})
execute_process(COMMAND ${Python_EXECUTABLE} ${SIMPLER_ROOT}/cmake/detect_torch_adapter.py OUTPUT_VARIABLE metadata OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" fields "${metadata}")
list(GET fields 0 torch_root)
list(GET fields 1 npu_root)
list(GET fields 2 torch_abi)
add_library(kernel_adapter_test_runtime SHARED runtime.cpp)
target_include_directories(kernel_adapter_test_runtime PRIVATE
    ${SIMPLER_ROOT}/src/common/worker ${SIMPLER_ROOT}/src/common/task_interface
    ${SIMPLER_ROOT}/src/common/log/include $ENV{ASCEND_HOME_PATH}/include)
target_link_directories(kernel_adapter_test_runtime PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(kernel_adapter_test_runtime PRIVATE ascendcl)
nanobind_add_module(_kernel_queue_probe queue_probe.cpp)
target_compile_definitions(_kernel_queue_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=${torch_abi})
target_include_directories(_kernel_queue_probe PRIVATE
    ${torch_root}/include ${torch_root}/include/torch/csrc/api/include
    ${npu_root}/include ${npu_root}/include/third_party/acl/inc ${npu_root}/include/third_party/hccl/inc)
target_link_libraries(_kernel_queue_probe PRIVATE
    ${torch_root}/lib/libtorch.so ${torch_root}/lib/libtorch_cpu.so
    ${torch_root}/lib/libtorch_python.so ${torch_root}/lib/libc10.so ${npu_root}/lib/libtorch_npu.so)
