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

#include <cstdint>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "common/platform_config.h"
#include "host/runtime_timeout_config.h"

namespace {

constexpr RuntimeTimeoutConfig kDefaults{
    PLATFORM_OP_EXECUTE_TIMEOUT_US, PLATFORM_STREAM_SYNC_TIMEOUT_MS, PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS
};
constexpr RuntimeTimeoutConfig kCiTightTimeouts{3000000, 4000, 2000};

void set_env_var(const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env_var(const char *name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedUnsetTimeoutEnv {
public:
    ScopedUnsetTimeoutEnv() {
        save(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, op_);
        save(SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, stream_);
        save(SIMPLER_SCHEDULER_TIMEOUT_MS_ENV, scheduler_);
        unset_env_var(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV);
        unset_env_var(SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV);
        unset_env_var(SIMPLER_SCHEDULER_TIMEOUT_MS_ENV);
    }

    ~ScopedUnsetTimeoutEnv() {
        restore(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, op_);
        restore(SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, stream_);
        restore(SIMPLER_SCHEDULER_TIMEOUT_MS_ENV, scheduler_);
    }

private:
    struct SavedValue {
        bool was_set{false};
        std::string value;
    };

    static void save(const char *name, SavedValue &out) {
        const char *value = std::getenv(name);
        if (value != nullptr) {
            out.was_set = true;
            out.value = value;
        }
    }

    static void restore(const char *name, const SavedValue &saved) {
        if (saved.was_set) {
            set_env_var(name, saved.value.c_str());
        } else {
            unset_env_var(name);
        }
    }

    SavedValue op_;
    SavedValue stream_;
    SavedValue scheduler_;
};

}  // namespace

TEST(RuntimeTimeoutConfig, UnsetEnvKeepsDefaults) {
    ScopedUnsetTimeoutEnv env;
    RuntimeTimeoutConfig cfg = resolve_runtime_timeout_config(kDefaults);

    EXPECT_EQ(cfg.op_execute_timeout_us, 45000000u);
    EXPECT_EQ(cfg.stream_sync_timeout_ms, 50000);
    EXPECT_EQ(cfg.scheduler_timeout_ms, 10000);
    EXPECT_EQ(validate_runtime_timeout_order(cfg), RuntimeTimeoutOrderStatus::OK);
}

TEST(RuntimeTimeoutConfig, ScopeStatsAppliesLargerDefaultFloor) {
    RuntimeTimeoutConfig cfg = apply_scope_stats_timeout_floor(kDefaults, true);

    EXPECT_EQ(cfg.op_execute_timeout_us, RUNTIME_TIMEOUT_SCOPE_STATS_OP_EXECUTE_US);
    EXPECT_EQ(cfg.stream_sync_timeout_ms, RUNTIME_TIMEOUT_SCOPE_STATS_STREAM_SYNC_MS);
    EXPECT_EQ(cfg.scheduler_timeout_ms, kDefaults.scheduler_timeout_ms);
    EXPECT_EQ(validate_runtime_timeout_order(cfg), RuntimeTimeoutOrderStatus::OK);
}

TEST(RuntimeTimeoutConfig, ScopeStatsDisabledKeepsDefaults) {
    RuntimeTimeoutConfig cfg = apply_scope_stats_timeout_floor(kDefaults, false);

    EXPECT_EQ(cfg.op_execute_timeout_us, kDefaults.op_execute_timeout_us);
    EXPECT_EQ(cfg.stream_sync_timeout_ms, kDefaults.stream_sync_timeout_ms);
    EXPECT_EQ(cfg.scheduler_timeout_ms, kDefaults.scheduler_timeout_ms);
}

TEST(RuntimeTimeoutConfig, ScopeStatsDoesNotLowerExistingBudgets) {
    constexpr RuntimeTimeoutConfig larger{120000000, 125000, 10000};
    RuntimeTimeoutConfig cfg = apply_scope_stats_timeout_floor(larger, true);

    EXPECT_EQ(cfg.op_execute_timeout_us, larger.op_execute_timeout_us);
    EXPECT_EQ(cfg.stream_sync_timeout_ms, larger.stream_sync_timeout_ms);
}

TEST(RuntimeTimeoutConfig, ScopeStatsFloorCanReturnToProductionDefaults) {
    RuntimeTimeoutConfig scope_cfg = apply_scope_stats_timeout_floor(kDefaults, true);
    RuntimeTimeoutConfig normal_cfg = apply_scope_stats_timeout_floor(kDefaults, false);

    EXPECT_EQ(scope_cfg.op_execute_timeout_us, RUNTIME_TIMEOUT_SCOPE_STATS_OP_EXECUTE_US);
    EXPECT_EQ(scope_cfg.stream_sync_timeout_ms, RUNTIME_TIMEOUT_SCOPE_STATS_STREAM_SYNC_MS);
    EXPECT_EQ(normal_cfg.op_execute_timeout_us, kDefaults.op_execute_timeout_us);
    EXPECT_EQ(normal_cfg.stream_sync_timeout_ms, kDefaults.stream_sync_timeout_ms);
}

TEST(RuntimeTimeoutConfig, ExplicitEnvOverridesScopeStatsDefaults) {
    ScopedUnsetTimeoutEnv env;
    set_env_var(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, "60000000");
    set_env_var(SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, "65000");
    RuntimeTimeoutConfig scope_defaults = apply_scope_stats_timeout_floor(kDefaults, true);
    RuntimeTimeoutConfig cfg = resolve_runtime_timeout_config(scope_defaults);

    EXPECT_EQ(cfg.op_execute_timeout_us, 60000000u);
    EXPECT_EQ(cfg.stream_sync_timeout_ms, 65000);
    EXPECT_EQ(validate_runtime_timeout_order(cfg), RuntimeTimeoutOrderStatus::OK);
}

TEST(RuntimeTimeoutConfig, ValidEnvOverridesDefaults) {
    ScopedUnsetTimeoutEnv env;
    set_env_var(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, "5000000");
    set_env_var(SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, "7000");
    set_env_var(SIMPLER_SCHEDULER_TIMEOUT_MS_ENV, "3000");

    RuntimeTimeoutParseStatus status;
    RuntimeTimeoutConfig cfg = resolve_runtime_timeout_config(kDefaults, &status);

    EXPECT_EQ(cfg.op_execute_timeout_us, 5000000u);
    EXPECT_EQ(cfg.stream_sync_timeout_ms, 7000);
    EXPECT_EQ(cfg.scheduler_timeout_ms, 3000);
    EXPECT_TRUE(status.op_execute_env_set);
    EXPECT_TRUE(status.stream_sync_env_set);
    EXPECT_TRUE(status.scheduler_env_set);
    EXPECT_TRUE(status.op_execute_valid);
    EXPECT_TRUE(status.stream_sync_valid);
    EXPECT_TRUE(status.scheduler_valid);
    EXPECT_EQ(validate_runtime_timeout_order(cfg), RuntimeTimeoutOrderStatus::OK);
}

TEST(RuntimeTimeoutConfig, InvalidEnvKeepsDefaultAndReportsStatus) {
    ScopedUnsetTimeoutEnv env;
    set_env_var(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, "12ms");

    RuntimeTimeoutParseStatus status;
    RuntimeTimeoutConfig cfg = resolve_runtime_timeout_config(kDefaults, &status);

    EXPECT_EQ(cfg.op_execute_timeout_us, 45000000u);
    EXPECT_TRUE(status.op_execute_env_set);
    EXPECT_FALSE(status.op_execute_valid);
}

TEST(RuntimeTimeoutConfig, ReusedParseStatusStartsClean) {
    ScopedUnsetTimeoutEnv env;
    RuntimeTimeoutParseStatus status;

    set_env_var(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, "5000000");
    resolve_runtime_timeout_config(kDefaults, &status);
    EXPECT_TRUE(status.op_execute_env_set);
    EXPECT_TRUE(status.op_execute_valid);

    unset_env_var(SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV);
    resolve_runtime_timeout_config(kDefaults, &status);
    EXPECT_FALSE(status.op_execute_env_set);
    EXPECT_TRUE(status.op_execute_valid);
}

TEST(RuntimeTimeoutConfig, CiTightTimeoutsStillValidate) {
    EXPECT_EQ(validate_runtime_timeout_order(kCiTightTimeouts), RuntimeTimeoutOrderStatus::OK);
}

TEST(RuntimeTimeoutConfig, InvalidTokenKeepsPriorValue) {
    uint64_t value = 42;

    EXPECT_FALSE(apply_runtime_timeout_override("SIMPLER_OP_EXECUTE_TIMEOUT_US", "12ms", 1, UINT64_MAX, &value));
    EXPECT_EQ(value, 42u);
}

TEST(RuntimeTimeoutConfig, RejectsBrokenOrdering) {
    RuntimeTimeoutConfig cfg = kDefaults;

    cfg.scheduler_timeout_ms = 46000;
    EXPECT_EQ(validate_runtime_timeout_order(cfg), RuntimeTimeoutOrderStatus::SCHEDULER_NOT_BELOW_OP_EXECUTE);

    cfg = kDefaults;
    cfg.op_execute_timeout_us = 55000000;
    EXPECT_EQ(validate_runtime_timeout_order(cfg), RuntimeTimeoutOrderStatus::OP_EXECUTE_NOT_BELOW_STREAM_SYNC);

    cfg = kDefaults;
    cfg.scheduler_timeout_ms = 44000;
    cfg.stream_sync_timeout_ms = 45100;
    EXPECT_EQ(validate_runtime_timeout_order(cfg), RuntimeTimeoutOrderStatus::STREAM_SYNC_NOT_COVERING_SCHEDULER_GUARD);
}

TEST(RuntimeTimeoutConfig, SimPlatformSkipsOnboardOrdering) {
    RuntimeTimeoutConfig cfg = kDefaults;
    cfg.scheduler_timeout_ms = 46000;

    EXPECT_EQ(validate_runtime_timeout_order_for_platform(cfg, "a2a3sim"), RuntimeTimeoutOrderStatus::OK);
    EXPECT_EQ(validate_runtime_timeout_order_for_platform(cfg, "a5sim"), RuntimeTimeoutOrderStatus::OK);
    EXPECT_EQ(
        validate_runtime_timeout_order_for_platform(cfg, "a2a3"),
        RuntimeTimeoutOrderStatus::SCHEDULER_NOT_BELOW_OP_EXECUTE
    );
    EXPECT_EQ(
        validate_runtime_timeout_order_for_platform(cfg, "a5"),
        RuntimeTimeoutOrderStatus::SCHEDULER_NOT_BELOW_OP_EXECUTE
    );
}
