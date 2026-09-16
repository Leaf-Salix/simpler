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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../protocol.h"
#include "load_aicpu_op.h"
#include "host/kernel_launch_binder.h"
#include "kernel_launch_sequence.h"
#include "kernel_platform_ops.h"

static void check(int rc, const char *where) {
    if (rc != 0) {
        std::printf("FAIL %s rc=%d\n", where, rc);
        std::fflush(stdout);
        std::_Exit(1);
    }
}
static std::vector<char> read_file(const char *path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) check(-1, path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
int main(int argc, char **argv) {
    if (argc != 7 || (std::string(argv[4]) != "eager" && std::string(argv[4]) != "replay") ||
        (std::string(argv[5]) != "check" && std::string(argv[5]) != "no-check") ||
        (std::string(argv[6]) != "success" && std::string(argv[6]) != "error")) {
        std::fprintf(stderr, "usage: probe DEVICE DISPATCHER_SO PROBE_SO eager|replay check|no-check success|error\n");
        return 1;
    }
    const bool replay = std::string(argv[4]) == "replay";
    const bool checked = std::string(argv[5]) == "check";
    const bool error = std::string(argv[6]) == "error";
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(std::atoi(argv[1])), "setDevice");
    aclrtStream caller{};
    check(aclrtCreateStream(&caller), "caller");
    KernelExecutionState state;
    check(state.initialize(std::atoi(argv[1]), make_onboard_kernel_context_ops(), 1), "K2 init");
    auto *cpu = state.hidden_stream(KernelStreamKind::Aicpu);
    auto *core = state.hidden_stream(KernelStreamKind::Aicore);
    auto dispatcher = read_file(argv[2]);
    auto library = read_file(argv[3]);
    // Intentionally process-lifetime: terminal native error is never followed
    // by destructor-based resource freeing, reset, or context reuse.
    auto *loader = new host::LoadAicpuOp;
    check(
        loader->BootstrapDispatcher(
            dispatcher.data(), dispatcher.size(), library.data(), library.size(), cpu, std::atoi(argv[1])
        ),
        "bootstrap"
    );
    check(loader->Init({"caller_probe_check"}), "load");
    void *device_report{}, *marker{}, *host_report{};
    check(aclrtMalloc(&device_report, sizeof(CallerProbeReport), ACL_MEM_MALLOC_NORMAL_ONLY), "report");
    check(aclrtMalloc(&marker, 64, ACL_MEM_MALLOC_NORMAL_ONLY), "marker");
    check(aclrtMallocHost(&host_report, sizeof(CallerProbeReport)), "pinned report");
    check(aclrtMemset(device_report, sizeof(CallerProbeReport), 0, sizeof(CallerProbeReport)), "zero report");
    check(aclrtSynchronizeStreamWithTimeout(cpu, 10000), "bootstrap completion");
    namespace kl = simpler::kernel_launch;
    kl::KernelLaunchHandles h{};
    h.caller = caller;
    h.aicpu = cpu;
    h.aicore = core;
    h.start = state.event(KernelEventKind::Start);
    h.aicore_start = state.event(KernelEventKind::AicoreStart);
    h.aicore_done = state.event(KernelEventKind::AicoreDone);
    h.aicpu_done = state.event(KernelEventKind::AicpuDone);
    h.serial_tail = state.event(KernelEventKind::SerialTail);
    struct Submission {
        host::LoadAicpuOp *loader;
        CallerProbeArgs args;
        void *marker;
        void *tail;
        bool checked;
    } submission{
        loader, {reinterpret_cast<uint64_t>(device_report), error ? -47 : 0, 0}, marker, h.serial_tail, checked
    };
    kl::KernelLaunchOps ops{};
    ops.context = &submission;
    ops.wait_event = [](void *, void *stream, void *event) noexcept {
        return aclrtStreamWaitEvent(stream, event);
    };
    ops.record_event = [](void *p, void *event, void *stream) noexcept {
        auto &s = *static_cast<Submission *>(p);
        // Candidate insertion point: actual sequence's JoinAicpu -> check ->
        // SerialTail, with the original hidden work still returning native=2.
        if (event == s.tail && s.checked) {
            const int rc = s.loader->LaunchBuiltInOp(stream, &s.args, sizeof(s.args), 1, "caller_probe_check");
            if (rc != 0) return rc;
        }
        return static_cast<int>(aclrtRecordEvent(event, stream));
    };
    ops.memset_handshake = [](void *p, void *stream) noexcept {
        auto &s = *static_cast<Submission *>(p);
        return static_cast<int>(aclrtMemsetAsync(
            reinterpret_cast<void *>(s.args.report), sizeof(CallerProbeReport), 0, sizeof(CallerProbeReport), stream
        ));
    };
    ops.launch_aicore = [](void *p, void *stream) noexcept {
        auto &s = *static_cast<Submission *>(p);
        // This is a transport/event probe, not AICore retirement acceptance.
        return static_cast<int>(aclrtMemsetAsync(s.marker, 64, 0, 64, stream));
    };
    ops.launch_aicpu = [](void *p, void *stream) noexcept {
        auto &s = *static_cast<Submission *>(p);
        return s.loader->LaunchBuiltInOp(stream, &s.args, sizeof(s.args), 1, host::KernelNames::RunName);
    };
    ops.cancel_waiting_aicore = [](void *, void *) noexcept {
        return 0;
    };
    aclmdlRI model{};
    if (replay) check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture begin");
    const auto submitted = kl::enqueue_kernel_launch_sequence(ops, h);
    std::printf(
        "enqueue rc=%d step=%d tail=%d\n", submitted.status, static_cast<int>(submitted.failed_step),
        submitted.tail_recorded
    );
    if (replay) {
        check(submitted.status, "capture enqueue");
        check(aclmdlRICaptureEnd(caller, &model), "capture end");
        const int rc = aclmdlRIExecuteAsync(model, caller);
        std::printf("replay enqueue=%d\n", rc);
        check(rc, "replay enqueue");
    }
    const int caller_rc = aclrtSynchronizeStreamWithTimeout(caller, 10000);
    const int cpu_rc = aclrtSynchronizeStreamWithTimeout(cpu, 10000);
    const int core_rc = aclrtSynchronizeStreamWithTimeout(core, 10000);
    const int copy_rc = aclrtMemcpyAsync(
        host_report, sizeof(CallerProbeReport), device_report, sizeof(CallerProbeReport), ACL_MEMCPY_DEVICE_TO_HOST,
        core
    );
    const int copy_sync = copy_rc == 0 ? aclrtSynchronizeStreamWithTimeout(core, 1000) : copy_rc;
    std::printf("caller=%d cpu=%d core=%d copy=%d copy_sync=%d\n", caller_rc, cpu_rc, core_rc, copy_rc, copy_sync);
    bool evidence = submitted.status == 0 && submitted.tail_recorded && copy_rc == 0 && copy_sync == 0;
    if (evidence) {
        const auto &r = *static_cast<const CallerProbeReport *>(host_report);
        std::printf(
            "work_complete=%lu work_status=%d check_complete=%lu observed=%d\n", r.work_complete, r.work_status,
            r.check_complete, r.observed_status
        );
        evidence = r.work_complete == 1 && r.work_status == (error ? -47 : 0) &&
                   (!checked || (r.check_complete == 1 && r.observed_status == r.work_status));
    }
    // Nonzero/timeout alone is not proof of the expected error.
    const bool expected_error =
        replay ? caller_rc == ACL_ERROR_RT_MODEL_EXECUTE : caller_rc == ACL_ERROR_RT_AICPU_EXCEPTION;
    const bool pass = evidence && (error ? expected_error : (caller_rc == 0 && cpu_rc == 0 && core_rc == 0));
    std::printf(
        "caller_probe %s replay=%d check=%d error=%d evidence=%d\n", pass ? "PASS" : "FAIL_OR_UNVERIFIED", replay,
        checked, error, evidence
    );
    std::fflush(stdout);
    std::_Exit(pass ? 0 : 1);
}
