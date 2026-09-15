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
#include <acl/error_codes/rt_error_codes.h>
#include <pthread.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <thread>

namespace {
void *resolve(const char *name) {
    if (auto *p = dlsym(RTLD_NEXT, name)) return p;
    for (auto *library : {"libascendcl.so", "libruntime.so"}) {
        auto *handle = dlopen(library, RTLD_NOLOAD | RTLD_NOW);
        if (!handle) continue;
        auto *p = dlsym(handle, name);
        dlclose(handle);
        if (p) return p;
    }
    std::fprintf(stderr, "cannot resolve %s\n", name);
    std::abort();
}
using Query = decltype(&aclrtQueryEventStatus);
Query real_query() {
    static auto fn = reinterpret_cast<Query>(resolve("aclrtQueryEventStatus"));
    return fn;
}
bool arm_record = false;
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool subscribed = false, entered = false, released = false, timed_out = false;
    std::atomic<bool> stop{false};
    aclError subscribe_rc = 0, process_rc = 0;
    aclrtStream stream = nullptr;
    aclrtEvent event = nullptr;
    uint64_t tid = 0;
    std::thread reporter;
};
Gate gate;

void block(void *) {
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.entered = true;
    gate.changed.notify_all();
    if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [] {
            return gate.released;
        })) {
        gate.timed_out = true;
        gate.released = true;
    }
}

aclError install_gate(aclrtStream stream) {
    aclrtContext context{};
    auto rc = aclrtGetCurrentContext(&context);
    if (rc) return rc;
    gate.stream = stream;
    gate.subscribed = gate.entered = gate.released = gate.timed_out = false;
    gate.subscribe_rc = gate.process_rc = 0;
    gate.stop = false;
    gate.reporter = std::thread([context] {
        auto status = aclrtSetCurrentContext(context);
        gate.tid = static_cast<uint64_t>(pthread_self());
        if (!status) status = aclrtSubscribeReport(gate.tid, gate.stream);
        {
            std::lock_guard<std::mutex> lock(gate.mutex);
            gate.subscribe_rc = status;
            gate.subscribed = true;
        }
        gate.changed.notify_all();
        while (!status && !gate.stop.load()) {
            auto processed = aclrtProcessReport(200);
            if (processed && processed != ACL_ERROR_RT_REPORT_TIMEOUT) {
                gate.process_rc = processed;
                break;
            }
        }
    });
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        if (!gate.changed.wait_for(lock, std::chrono::seconds(10), [] {
                return gate.subscribed;
            }))
            return -4904;
    }
    if (gate.subscribe_rc) return gate.subscribe_rc;
    rc = aclrtLaunchCallback(block, nullptr, ACL_CALLBACK_BLOCK, stream);
    if (rc) return rc;
    std::unique_lock<std::mutex> lock(gate.mutex);
    if (!gate.changed.wait_for(lock, std::chrono::seconds(5), [] {
            return gate.entered;
        }))
        return -4901;
    return 0;
}
}  // namespace

extern "C" void capture_gate_arm() { arm_record = true; }
extern "C" int capture_gate_pending() {
    if (arm_record || !gate.event) return -4902;
    aclrtEventRecordedStatus status{};
    auto rc = real_query()(gate.event, &status);
    return rc ? -4903 : status == ACL_EVENT_RECORDED_STATUS_NOT_READY;
}
extern "C" int capture_gate_blocked() {
    std::lock_guard<std::mutex> lock(gate.mutex);
    return gate.entered && !gate.released && !gate.timed_out;
}
extern "C" void capture_gate_release() {
    {
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.released = true;
    }
    gate.changed.notify_all();
}
extern "C" int capture_gate_finish() {
    capture_gate_release();
    const auto rc = aclrtSynchronizeStreamWithTimeout(gate.stream, 10000);
    if (rc) std::_Exit(4);
    gate.stop = true;
    if (gate.reporter.joinable()) gate.reporter.join();
    auto unsubscribed = aclrtUnSubscribeReport(gate.tid, gate.stream);
    return gate.process_rc ? gate.process_rc : unsubscribed;
}

extern "C" aclError capture_gate_before_record(aclrtEvent event, aclrtStream stream) {
    if (!arm_record) return 0;
    arm_record = false;
    gate.event = event;
    return install_gate(stream);
}
