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

#include <nanobind/nanobind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/Functions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include "recording_allocator.h"

namespace nb = nanobind;
namespace {
std::mutex mutex;
std::condition_variable changed;
bool started = false;
bool released = false;
void hold() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        started = false;
        released = false;
    }
    at_npu::native::OpCommand::RunOpApiV2(
        "kernel_adapter_test_hold",
        [] {
            std::unique_lock<std::mutex> lock(mutex);
            started = true;
            changed.notify_all();
            if (!changed.wait_for(lock, std::chrono::seconds(30), [] {
                    return released;
                }))
                return static_cast<int>(ACL_ERROR_INTERNAL_ERROR);
            return static_cast<int>(ACL_SUCCESS);
        },
        false
    );
}
void wait_started() {
    std::unique_lock<std::mutex> lock(mutex);
    if (!changed.wait_for(lock, std::chrono::seconds(10), [] {
            return started;
        }))
        throw std::runtime_error("Torch taskQueue callback did not start");
}
void release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    changed.notify_all();
}

nb::object external_alias(nb::handle value) {
    if (!THPVariable_Check(value.ptr())) throw nb::type_error("expected a Tensor");
    const auto &source = THPVariable_Unpack(value.ptr());
    auto result = at::empty_strided(source.sizes(), source.strides(), source.options());
    c10::DataPtr ptr(
        source.storage().data_ptr().get(), new at::Tensor(source),
        [](void *p) {
            delete static_cast<at::Tensor *>(p);
        },
        source.device()
    );
    result.storage().set_data_ptr(std::move(ptr));
    result.storage().set_nbytes(source.storage().nbytes());
    result.unsafeGetTensorImpl()->set_storage_offset(source.storage_offset());
    return nb::steal<nb::object>(THPVariable_Wrap(result));
}
}  // namespace
NB_MODULE(_kernel_queue_probe, m) {
    m.def("hold", &hold, nb::call_guard<nb::gil_scoped_release>());
    m.def("wait_started", &wait_started, nb::call_guard<nb::gil_scoped_release>());
    m.def("release", &release, nb::call_guard<nb::gil_scoped_release>());
    m.def("external_alias", &external_alias);
    m.def("start_recording", [] {
        kernel_adapter_test::recording_allocator().start();
    });
    m.def("stop_recording", [] {
        kernel_adapter_test::recording_allocator().stop();
    });
    m.def("record_count", [](uintptr_t addr) {
        return kernel_adapter_test::recording_allocator().count(addr);
    });
    m.def("allocation_base", [](uintptr_t addr) {
        return reinterpret_cast<uintptr_t>(
            c10_npu::NPUCachingAllocator::getBaseAllocation(reinterpret_cast<void *>(addr), nullptr)
        );
    });
}
