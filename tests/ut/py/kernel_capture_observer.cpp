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

#include <acl/acl.h>
#include <runtime/rt.h>
#include <runtime/rts/rts_kernel.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>

#include "common/kernel_args.h"
#include "task_interface/kernel_dispatch_args.h"
#include "tensormap_and_ringbuffer/kernel_invocation.h"

namespace {
enum class ObserverError : int {
    None,
    InvalidCoreArgs,
    InvalidCpuArgs,
    BindingChanged,
    NoPairedLaunches,
    InvalidResident,
    ResidentChanged,
};

// The isolated test has one submitting host thread and one prepared context.
struct Observer {
    bool armed{false};
    bool resident_sampled{false};
    ObserverError error{ObserverError::None};
    uint64_t core_launches{0};
    uint64_t cpu_launches{0};
    uint64_t binding{0};
    uint64_t context_generation{0};
    KernelArgs resident{};
};
Observer observer;

void note_error(ObserverError error) {
    if (observer.error == ObserverError::None) observer.error = error;
}

void note_binding(uint64_t binding) {
    if (binding == 0 || (observer.binding != 0 && observer.binding != binding)) {
        note_error(ObserverError::BindingChanged);
    } else {
        observer.binding = binding;
    }
}

void *resolve_cann_symbol(const char *symbol) {
    if (void *address = dlsym(RTLD_NEXT, symbol)) return address;
    // The ctypes runtime loader keeps CANN dependencies in RTLD_LOCAL scope.
    for (const char *library : {"libascendcl.so", "libruntime.so"}) {
        void *handle = dlopen(library, RTLD_NOLOAD | RTLD_NOW);
        if (handle == nullptr) continue;
        void *address = dlsym(handle, symbol);
        dlclose(handle);
        if (address != nullptr) return address;
    }
    std::fprintf(stderr, "kernel_capture_observer: cannot resolve %s\n", symbol);
    return nullptr;
}

void observe_core(const rtArgsEx_t *args) {
    ++observer.core_launches;
    if (args == nullptr || args->args == nullptr || args->argsSize != sizeof(KernelArgs *)) {
        note_error(ObserverError::InvalidCoreArgs);
        return;
    }
    KernelArgs *device_args = nullptr;
    std::memcpy(&device_args, args->args, sizeof(device_args));
    note_binding(reinterpret_cast<uintptr_t>(device_args));
}

void observe_cpu(const rtCpuKernelArgs_t *args) {
    ++observer.cpu_launches;
    using simpler::tmr::TmrBindingRef;
    if (args == nullptr || args->baseArgs.args == nullptr ||
        args->baseArgs.argsSize < sizeof(SimplerKernelDispatchArgs) + sizeof(TmrBindingRef)) {
        note_error(ObserverError::InvalidCpuArgs);
        return;
    }
    const auto *bytes = static_cast<const uint8_t *>(args->baseArgs.args);
    SimplerKernelDispatchArgs prefix{};
    TmrBindingRef reference{};
    std::memcpy(&prefix, bytes, sizeof(prefix));
    std::memcpy(&reference, bytes + sizeof(prefix), sizeof(reference));
    if (prefix.packet_bytes != args->baseArgs.argsSize || prefix.invocation.mode != SIMPLER_MODE_KERNEL ||
        prefix.invocation.payload_bytes != args->baseArgs.argsSize - sizeof(prefix) ||
        prefix.binding_address != reference.device_binding_addr || prefix.context_generation == 0 ||
        prefix.context_generation != reference.context_generation) {
        note_error(ObserverError::InvalidCpuArgs);
        return;
    }
    note_binding(prefix.binding_address);
    if (observer.context_generation != 0 && observer.context_generation != prefix.context_generation)
        note_error(ObserverError::BindingChanged);
    observer.context_generation = prefix.context_generation;
}
}  // namespace

extern "C" void capture_observer_begin() {
    observer = {};
    observer.armed = true;
}

extern "C" int capture_observer_status() {
    if (observer.error != ObserverError::None) return static_cast<int>(observer.error);
    if (!observer.armed || observer.core_launches == 0 || observer.core_launches != observer.cpu_launches)
        return static_cast<int>(ObserverError::NoPairedLaunches);
    return 0;
}

extern "C" uint64_t capture_observer_core_launches() { return observer.core_launches; }
extern "C" uint64_t capture_observer_cpu_launches() { return observer.cpu_launches; }
extern "C" uint64_t capture_observer_kernel_args() { return observer.binding; }
extern "C" uint64_t capture_observer_runtime_args() {
    return reinterpret_cast<uintptr_t>(observer.resident.runtime_args);
}
extern "C" uint64_t capture_observer_regs() { return observer.resident.regs; }

// The caller establishes quiescence outside capture before this D2H read.
extern "C" int capture_observer_check_resident() {
    const int status = capture_observer_status();
    if (status != 0) return status;
    const auto copy = reinterpret_cast<decltype(&aclrtMemcpy)>(resolve_cann_symbol("aclrtMemcpy"));
    if (copy == nullptr) return -4330;
    KernelArgs resident{};
    const int rc = copy(
        &resident, sizeof(resident), reinterpret_cast<const void *>(observer.binding), sizeof(resident),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    if (rc != 0) return rc;
    if (resident.runtime_args == nullptr || resident.regs == 0) {
        note_error(ObserverError::InvalidResident);
    } else if (observer.resident_sampled &&
               (resident.runtime_args != observer.resident.runtime_args || resident.regs != observer.resident.regs ||
                resident.ffts_base_addr != observer.resident.ffts_base_addr)) {
        note_error(ObserverError::ResidentChanged);
    } else {
        observer.resident = resident;
        observer.resident_sampled = true;
    }
    return static_cast<int>(observer.error);
}

extern "C" rtError_t rtKernelLaunchWithHandleV2(
    void *handle, const uint64_t tiling_key, uint32_t blocks, rtArgsEx_t *args, rtSmDesc_t *sm_desc, rtStream_t stream,
    const rtTaskCfgInfo_t *config
) {
    if (observer.armed) observe_core(args);
    static const auto real =
        reinterpret_cast<decltype(&rtKernelLaunchWithHandleV2)>(resolve_cann_symbol("rtKernelLaunchWithHandleV2"));
    return real == nullptr ? -4330 : real(handle, tiling_key, blocks, args, sm_desc, stream, config);
}

extern "C" rtError_t rtsLaunchCpuKernel(
    const rtFuncHandle function, uint32_t blocks, rtStream_t stream, const rtKernelLaunchCfg_t *config,
    rtCpuKernelArgs_t *args
) {
    if (observer.armed) observe_cpu(args);
    static const auto real = reinterpret_cast<decltype(&rtsLaunchCpuKernel)>(resolve_cann_symbol("rtsLaunchCpuKernel"));
    return real == nullptr ? -4330 : real(function, blocks, stream, config, args);
}
