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
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include "kernel_host_owner.h"
#include "kernel_queue_bridge.h"

namespace nb = nanobind;
using simpler::kernel::KernelHostOwner;
using simpler::kernel::KernelQueueBridge;
using simpler::kernel::KernelQueuedInvocation;
using simpler::kernel::KernelQueueTicket;

namespace {
struct BridgeOwner {
    std::shared_ptr<KernelHostOwner> owner;
    KernelQueueBridge api;
};
int accept_call(
    void *opaque, int32_t cid, const ChipTensor *tensors, size_t nt, const uint64_t *scalars, size_t ns,
    KernelQueueTicket *out
) noexcept {
    try {
        auto &owner = static_cast<BridgeOwner *>(opaque)->owner;
        auto call = owner->accept(cid, tensors, nt, scalars, ns);
        auto storage = new std::shared_ptr<KernelQueuedInvocation>(std::move(call));
        *out = {
            storage,
            [](void *p) noexcept {
                delete static_cast<std::shared_ptr<KernelQueuedInvocation> *>(p);
            },
            [](void *p, void *stream, int (*before)(void *) noexcept, void *data) noexcept {
                return (*static_cast<std::shared_ptr<KernelQueuedInvocation> *>(p))->invoke(stream, before, data);
            }
        };
        return 0;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}
std::vector<uint8_t> copy_bytes(nb::bytes bytes) {
    const auto *p = reinterpret_cast<const uint8_t *>(bytes.c_str());
    return {p, p + bytes.size()};
}
}  // namespace

nb::capsule make_kernel_queue_handle(std::shared_ptr<KernelHostOwner> owner) {
    int device;
    {
        nb::gil_scoped_release release;
        device = owner->device();
    }
    auto bridge = std::make_unique<BridgeOwner>();
    bridge->owner = std::move(owner);
    bridge->api = {1, sizeof(KernelQueueBridge), device, bridge.get(), &accept_call};
    nb::capsule result(&bridge->api, simpler::kernel::KERNEL_QUEUE_CAPSULE, [](void *p) noexcept {
        auto api = static_cast<KernelQueueBridge *>(p);
        delete static_cast<BridgeOwner *>(api->opaque);
    });
    bridge.release();
    return result;
}

void bind_kernel_host(nb::module_ &m) {
    m.def(
        "create_kernel_host_owner",
        [](const std::string &path) {
            return std::make_shared<KernelHostOwner>(path);
        },
        nb::arg("runtime_path"), nb::call_guard<nb::gil_scoped_release>()
    );
    nb::class_<KernelHostOwner>(m, "KernelHostOwner")
        .def(
            "initialize",
            [](KernelHostOwner &owner, int device, nb::bytes cpu, nb::bytes core, nb::bytes dispatcher,
               const CallConfig &config, uint64_t generation) {
                auto cpu_copy = copy_bytes(cpu), core_copy = copy_bytes(core), dispatcher_copy = copy_bytes(dispatcher);
                const CallConfig config_copy = config;
                nb::gil_scoped_release release;
                return owner.initialize(device, cpu_copy, core_copy, dispatcher_copy, config_copy, generation);
            },
            nb::arg("device"), nb::arg("aicpu_binary"), nb::arg("aicore_binary"), nb::arg("dispatcher_binary"),
            nb::arg("config"), nb::arg("context_generation")
        )
        .def(
            "prepare",
            [](KernelHostOwner &owner, int32_t cid, nb::bytes callable) {
                auto copy = copy_bytes(callable);
                nb::gil_scoped_release release;
                return owner.prepare(cid, copy.data(), copy.size());
            },
            nb::arg("callable_id"), nb::arg("callable")
        )
        .def(
            "close", &KernelHostOwner::close, nb::call_guard<nb::gil_scoped_release>(),
            "Explicit close. Requires Device quiescence and destruction of all referencing graphs."
        )
        .def("supported", &KernelHostOwner::supported, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("closed", &KernelHostOwner::closed, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("pending", &KernelHostOwner::pending, nb::call_guard<nb::gil_scoped_release>())
        .def("_queue_handle", &make_kernel_queue_handle);
}
