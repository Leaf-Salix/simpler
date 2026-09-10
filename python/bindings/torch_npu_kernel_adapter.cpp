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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <cstdio>
#include <limits>
#include <memory>
#include <unordered_set>
#include "kernel_queue_bridge.h"
#include "runtime_c_api.h"
#include "kernel_invocation_validation.h"
#include "torch_adapter_build.h"

namespace nb = nanobind;
using namespace simpler::kernel;
namespace {
DataType dtype_of(at::ScalarType type) {
    switch (type) {
    case at::kFloat:
        return DataType::FLOAT32;
    case at::kHalf:
        return DataType::FLOAT16;
    case at::kBFloat16:
        return DataType::BFLOAT16;
    case at::kInt:
        return DataType::INT32;
    case at::kShort:
        return DataType::INT16;
    case at::kChar:
        return DataType::INT8;
    case at::kByte:
        return DataType::UINT8;
    case at::kLong:
        return DataType::INT64;
    case at::kBool:
        return DataType::BOOL;
    default:
        throw std::invalid_argument("unsupported kernel Tensor dtype");
    }
}

ChipTensor describe(const at::Tensor &tensor, int device) {
    if (!tensor.defined() || tensor.device().type() != c10::DeviceType::PrivateUse1 || tensor.get_device() != device ||
        tensor.layout() != c10::kStrided)
        throw std::invalid_argument("kernel Tensor must be strided NPU storage on the owner device");
    if (tensor.requires_grad())
        throw std::invalid_argument("kernel adapter is inference-only; detach tensors requiring gradients");
    if (tensor.is_neg() || tensor.is_conj())
        throw std::invalid_argument("kernel Tensor cannot have unresolved negative/conjugate view bits");
    const auto format = at_npu::native::get_npu_format(tensor);
    if (format != ACL_FORMAT_ND && format != ACL_FORMAT_NCHW && format != ACL_FORMAT_NCDHW)
        throw std::invalid_argument("kernel Tensor requires a linear ND/NCHW/NCDHW format");
    if (tensor.dim() <= 0 || tensor.dim() > MAX_TENSOR_DIMS)
        throw std::invalid_argument("kernel Tensor rank must be in [1, MAX_TENSOR_DIMS]");
    ChipTensor out{};
    out.dtype = dtype_of(tensor.scalar_type());
    out.address_space = AddressSpace::DEVICE;
    out.ndims = static_cast<uint32_t>(tensor.dim());
    out.buffer.addr = reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get());
    out.buffer.size = tensor.storage().nbytes();
    if (tensor.storage_offset() < 0) throw std::invalid_argument("negative Tensor storage offset");
    out.start_offset = static_cast<uint64_t>(tensor.storage_offset());
    const bool empty = tensor.numel() == 0;
    uint64_t end = out.start_offset;
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        auto shape = tensor.size(i), stride = tensor.stride(i);
        if (shape < 0 || shape > UINT32_MAX || stride < 0 || stride > UINT32_MAX || (!empty && stride == 0))
            throw std::invalid_argument("kernel Tensor shape/stride is not representable");
        out.shapes[i] = static_cast<uint32_t>(shape);
        out.strides[i] = static_cast<uint32_t>(stride);
        if (!empty) {
            const uint64_t extent = static_cast<uint64_t>(shape - 1) * static_cast<uint64_t>(stride);
            if (extent > UINT64_MAX - end) throw std::invalid_argument("kernel Tensor extent overflow");
            end += extent;
        }
    }
    const uint64_t elements = out.buffer.size / tensor.element_size();
    if ((!empty && (!out.buffer.addr || end >= elements)) || (empty && out.start_offset > elements) ||
        out.buffer.addr > UINT64_MAX - out.buffer.size)
        throw std::invalid_argument("kernel Tensor view exceeds storage");
    return out;
}

void check_allocator(const at::Tensor &tensor) {
    auto *allocator = c10_npu::NPUCachingAllocator::get();
    if (allocator->name() != "native")
        throw std::invalid_argument("kernel adapter requires the default caching allocator");
    const auto &ptr = tensor.storage().data_ptr();
    if (ptr.get() && ptr.get_deleter() != allocator->raw_deleter())
        throw std::invalid_argument("kernel adapter cannot retain external Tensor storage");
    // raw_deleter also changes in uncached mode. Require a live allocator
    // block as well; recordStream silently skips non-caching DataPtr deleters.
    if (ptr.get()) allocator->getBaseAllocation(ptr.get(), nullptr);
}

struct DeferredInvocation {
    KernelQueueTicket ticket{};
    std::vector<at::Tensor> tensors;
    std::vector<ChipTensor> descriptors;
    c10_npu::NPUStream stream;
    explicit DeferredInvocation(c10_npu::NPUStream value) :
        stream(value) {}
    ~DeferredInvocation() {
        if (ticket.opaque) ticket.release(ticket.opaque);
    }

    static int record(void *opaque) noexcept {
        auto &self = *static_cast<DeferredInvocation *>(opaque);
        try {
            std::unordered_set<void *> recorded;
            for (size_t i = 0; i < self.tensors.size(); ++i) {
                const auto &tensor = self.tensors[i];
                check_allocator(tensor);
                const auto &ptr = tensor.storage().data_ptr();
                if (reinterpret_cast<uintptr_t>(ptr.get()) != self.descriptors[i].buffer.addr ||
                    tensor.storage().nbytes() != self.descriptors[i].buffer.size)
                    throw std::runtime_error("kernel Tensor storage changed while queued");
                if (ptr.get() && recorded.insert(ptr.get()).second)
                    c10_npu::NPUCachingAllocator::recordStream(ptr, self.stream);
            }
            return 0;
        } catch (const std::exception &error) {
            std::fprintf(stderr, "kernel storage recording failed: %s\n", error.what());
            return PTO_RUNTIME_ERR_INTERNAL;
        } catch (...) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    }
};

void enqueue(
    nb::handle capsule, int32_t cid, nb::iterable inputs, const std::vector<uint64_t> &scalars, const std::string &name
) {
    if (!PyCapsule_IsValid(capsule.ptr(), KERNEL_QUEUE_CAPSULE))
        throw nb::type_error("expected a KernelHostOwner queue handle");
    const auto *bridge =
        static_cast<const KernelQueueBridge *>(PyCapsule_GetPointer(capsule.ptr(), KERNEL_QUEUE_CAPSULE));
    if (!bridge || bridge->version != 1 || bridge->struct_size != sizeof(KernelQueueBridge) || !bridge->opaque ||
        !bridge->accept || bridge->device < 0)
        throw nb::value_error("incompatible kernel queue bridge");
    if (name.empty()) throw nb::value_error("kernel op_name must not be empty");
    nb::list values = nb::steal<nb::list>(PySequence_List(inputs.ptr()));
    if (!values.is_valid()) throw nb::python_error();
    auto stream = c10_npu::getCurrentNPUStream();
    if (stream.device_index() != bridge->device)
        throw nb::value_error("current NPU stream does not match the kernel owner device");
    auto deferred = std::make_shared<DeferredInvocation>(stream);
    for (nb::handle value : values) {
        if (!THPVariable_Check(value.ptr())) throw nb::type_error("kernel inputs must contain Tensor objects");
        at::Tensor tensor = THPVariable_Unpack(value.ptr());
        deferred->descriptors.push_back(describe(tensor, bridge->device));
        check_allocator(tensor);
        deferred->tensors.push_back(std::move(tensor));
        if (deferred->tensors.size() > CHIP_MAX_TENSOR_ARGS) throw nb::value_error("too many kernel Tensor arguments");
    }
    if (scalars.size() > CHIP_MAX_SCALAR_ARGS ||
        !valid_invocation_counts(static_cast<int32_t>(deferred->tensors.size()), static_cast<int32_t>(scalars.size())))
        throw nb::value_error("invalid kernel Tensor/scalar counts");
    void *raw = stream.stream(false);
    if (!raw) throw nb::value_error("null caller stream");
    // The Python call arguments keep the capsule alive through accept. The
    // accepted ticket then owns its context lease independently of Python.
    nb::gil_scoped_release release;
    int rc = bridge->accept(
        bridge->opaque, cid, deferred->descriptors.data(), deferred->descriptors.size(), scalars.data(), scalars.size(),
        &deferred->ticket
    );
    if (rc != 0) throw std::runtime_error("kernel invocation was not accepted: " + std::to_string(rc));
    std::function<int()> callback = [deferred, raw]() noexcept {
        const int result =
            deferred->ticket.invoke(deferred->ticket.opaque, raw, &DeferredInvocation::record, deferred.get());
        if (result == 0) return static_cast<int>(ACL_SUCCESS);
        std::fprintf(stderr, "kernel native enqueue failed: status=%d\n", result);
        // Do not feed native failures to Torch's OOM recovery/retry loop.
        return static_cast<int>(ACL_ERROR_INTERNAL_ERROR);
    };
    at_npu::native::OpCommand::RunOpApiV2(name, callback, false);
}
}  // namespace
NB_MODULE(_torch_npu_adapter, m) {
    m.doc() = "Optional L2 kernel mode Torch-NPU adapter";
    m.attr("BUILD_TORCH_VERSION") = kTorchBuildVersion;
    m.attr("BUILD_TORCH_NPU_VERSION") = kTorchNpuBuildVersion;
    m.def(
        "enqueue", &enqueue, nb::arg("owner_handle"), nb::arg("callable_id"), nb::arg("tensors"),
        nb::arg("scalar_bits"), nb::arg("op_name")
    );
}
