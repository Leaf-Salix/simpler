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

#include "../protocol.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/thread_scheduling.h"

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *) { return 0; }
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *packet) {
    (void)use_normal_aicpu_scheduling();
    const auto &args = *static_cast<const CallerProbeArgs *>(packet);
    auto *report = reinterpret_cast<CallerProbeReport *>(args.report);
    report->work_status = args.status;
    report->work_complete = 1;
    cache_flush_range(report, 64);
    return args.status == 0 ? 0 : 2;
}
extern "C" __attribute__((visibility("default"))) int caller_probe_check(void *packet) {
    const auto &args = *static_cast<const CallerProbeArgs *>(packet);
    auto *report = reinterpret_cast<CallerProbeReport *>(args.report);
    cache_invalidate_range(report, 64);
    const int status = report->work_complete == 1 ? report->work_status : -91;
    report->observed_status = status;
    report->check_complete = 1;
    cache_flush_range(reinterpret_cast<unsigned char *>(report) + 64, 64);
    return status == 0 ? 0 : 2;
}
