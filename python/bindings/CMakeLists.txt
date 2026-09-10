# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# nanobind Python bindings for task_interface and distributed runtime

# Torch's private C++ ABI stays in a separate optional extension.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/detect_torch_adapter.py
        OUTPUT_VARIABLE _torch_metadata RESULT_VARIABLE _torch_status OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _torch_status EQUAL 0)
        message(FATAL_ERROR "Cannot inspect installed Torch adapter dependencies")
    endif()
    if(_torch_metadata)
        string(REPLACE "\n" ";" _torch_fields "${_torch_metadata}")
        list(GET _torch_fields 0 _torch_root)
        list(GET _torch_fields 1 _torch_npu_root)
        list(GET _torch_fields 2 _torch_abi)
        list(GET _torch_fields 3 TORCH_ADAPTER_VERSION)
        list(GET _torch_fields 4 TORCH_ADAPTER_NPU_VERSION)
        configure_file(torch_adapter_build.h.in torch_adapter_build.h @ONLY)
        nanobind_add_module(_torch_npu_adapter torch_npu_kernel_adapter.cpp)
        target_include_directories(_torch_npu_adapter PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_SOURCE_DIR}/src/common/worker
            ${CMAKE_SOURCE_DIR}/src/common/task_interface
            ${_torch_root}/include
            ${_torch_root}/include/torch/csrc/api/include
            ${_torch_npu_root}/include
            ${_torch_npu_root}/include/third_party/acl/inc
            ${_torch_npu_root}/include/third_party/hccl/inc)
        target_compile_definitions(_torch_npu_adapter PRIVATE _GLIBCXX_USE_CXX11_ABI=${_torch_abi})
        target_link_libraries(_torch_npu_adapter PRIVATE
            ${_torch_root}/lib/libtorch.so ${_torch_root}/lib/libtorch_cpu.so
            ${_torch_root}/lib/libtorch_python.so ${_torch_root}/lib/libc10.so
            ${_torch_npu_root}/lib/libtorch_npu.so)
        set_target_properties(_torch_npu_adapter PROPERTIES
            BUILD_RPATH "${_torch_root}/lib;${_torch_npu_root}/lib"
            INSTALL_RPATH "$ORIGIN/torch/lib;$ORIGIN/torch_npu/lib")
        if(SKBUILD_MODE)
            install(TARGETS _torch_npu_adapter DESTINATION .)
        else()
            set_target_properties(_torch_npu_adapter PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
        endif()
    else()
        message(STATUS "Torch-NPU not installed: building without the optional kernel adapter")
    endif()
endif()


set(BINDING_SOURCES
    task_interface.cpp
    kernel_host.cpp
)

list(TRANSFORM BINDING_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

set(HIERARCHICAL_SRC ${CMAKE_SOURCE_DIR}/src/common/hierarchical)

set(HIERARCHICAL_SOURCES
    ${HIERARCHICAL_SRC}/types.cpp
    ${HIERARCHICAL_SRC}/tensormap.cpp
    ${HIERARCHICAL_SRC}/ring.cpp
    ${HIERARCHICAL_SRC}/scope.cpp
    ${HIERARCHICAL_SRC}/remote_wire.cpp
    ${HIERARCHICAL_SRC}/remote_endpoint.cpp
    ${HIERARCHICAL_SRC}/orchestrator.cpp
    ${HIERARCHICAL_SRC}/worker_manager.cpp
    ${HIERARCHICAL_SRC}/scheduler.cpp
    ${HIERARCHICAL_SRC}/worker.cpp
)

# Stamp the source commit into the module so `simpler.task_interface` can refuse
# to run against a binding built from a different tree. Same notion of "which
# sources is this built from" as RuntimeBuilder._build_cache_stamp, and git for
# the same reason: mtimes do not survive a branch switch. Empty when git is
# unavailable (release tarball, no .git), which disables the check.
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SIMPLER_BUILD_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

nanobind_add_module(_task_interface ${BINDING_SOURCES} ${HIERARCHICAL_SOURCES})

target_compile_definitions(_task_interface PRIVATE
    SIMPLER_BUILD_COMMIT="${SIMPLER_BUILD_COMMIT}"
)

target_sources(_task_interface PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/worker/kernel_host_owner.cpp
    ${CMAKE_SOURCE_DIR}/src/common/worker/chip_worker.cpp
    ${CMAKE_SOURCE_DIR}/src/common/worker/chip_run_lane.cpp
    ${CMAKE_SOURCE_DIR}/src/common/log/host_log.cpp
    ${CMAKE_SOURCE_DIR}/src/common/log/unified_log_host.cpp
    # Host-side assert_impl / get_stacktrace for the unified Tensor's
    # always_assert (Tensor::make -> init_external). The binding does not link
    # the runtime orchestration/common.cpp that provides these for runtime .so.
    ${CMAKE_SOURCE_DIR}/src/common/task_interface/assert_compat.cpp
)

target_include_directories(_task_interface PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/task_interface
    ${CMAKE_SOURCE_DIR}/src/common/worker
    ${CMAKE_SOURCE_DIR}/src/common/hierarchical
    ${CMAKE_SOURCE_DIR}/src/common/platform/include
    ${CMAKE_SOURCE_DIR}/src/common/platform/include/common
    ${CMAKE_SOURCE_DIR}/src/common/platform/include/host
    ${CMAKE_SOURCE_DIR}/src/common/log/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_task_interface PRIVATE ${CMAKE_DL_LIBS} pthread)

# POSIX shm_open / shm_unlink (used by worker_manager.cpp broadcast_register) live
# in librt on glibc < 2.34 and in libSystem on macOS. glibc 2.34+ folded librt
# into libc, so -lrt is a no-op there but harmless (and required for older glibc).
if(UNIX AND NOT APPLE)
    target_link_libraries(_task_interface PRIVATE rt)
endif()

if(SKBUILD_MODE)
    install(TARGETS _task_interface DESTINATION .)
else()
    set_target_properties(_task_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    )
endif()
