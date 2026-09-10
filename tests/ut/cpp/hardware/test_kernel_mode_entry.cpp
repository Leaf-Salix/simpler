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

/*
 * Hardware UT for the kernel-mode entry layer on a2a3 onboard: the four C ABI
 * entries reached through a real libhost_runtime.so, on a device and stream the
 * test owns rather than simpler.
 *
 * The borrowing is the point. Every case here does its own aclInit +
 * aclrtSetDevice + aclrtCreateStream and hands that stream in, so what is under
 * test is the borrowed-device shape a kernel-mode caller presents. A test that
 * let simpler stand the device up would exercise the program-mode shape and
 * pass for the wrong reason.
 *
 * TMR initialization establishes its private streams without borrowing the
 * caller stream. Execution remains unsupported until the binding/executor
 * providers exist: supported is zero and launch returns INVALID_STATE.
 * Malformed arguments return INTERNAL before state admission. Finalization
 * releases the private context without invalidating the caller stream.
 *
 * Hardware classification: requires_hardware_a2a3 (ctest label) + CMake gate
 * SIMPLER_ENABLE_HARDWARE_TESTS. Device allocation is driven by CTest
 * RESOURCE_GROUPS + --resource-spec-file.
 *
 * PTO_HOST_RUNTIME_LIB_PATH and the three binary paths are baked in at
 * configure time by tests/ut/cpp/CMakeLists.txt.
 */

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "call_config.h"
#include "callable.h"
#include "callable_protocol.h"
// ChipWorker holds a unique_ptr<ChipRunLane>, whose deleter needs the complete
// type wherever a ChipWorker is destroyed.
#include "chip_run_lane.h"
#include "chip_worker.h"
#include "runtime_c_api.h"
#include "task_args.h"

namespace {

/// The four kernel-mode entries, resolved out of libhost_runtime.so the way
/// ChipWorker resolves them.
struct KernelModeApi {
    void *(*create_device_context)();
    void (*destroy_device_context)(void *);
    decltype(&simpler_kernel_mode_supported) supported;
    decltype(&simpler_kernel_mode_init) init;
    decltype(&simpler_kernel_mode_prepare_callable) prepare_callable;
    decltype(&simpler_kernel_mode_launch) launch;
    decltype(&finalize_device) finalize;
};

template <typename F>
F resolve(void *handle, const char *name) {
    dlerror();
    void *sym = dlsym(handle, name);
    if (dlerror() != nullptr) return nullptr;
    return reinterpret_cast<F>(sym);
}

bool load_kernel_mode_api(void *handle, KernelModeApi &api) {
    api.create_device_context = resolve<decltype(api.create_device_context)>(handle, "create_device_context");
    api.destroy_device_context = resolve<decltype(api.destroy_device_context)>(handle, "destroy_device_context");
    api.supported = resolve<decltype(api.supported)>(handle, "simpler_kernel_mode_supported");
    api.init = resolve<decltype(api.init)>(handle, "simpler_kernel_mode_init");
    api.prepare_callable = resolve<decltype(api.prepare_callable)>(handle, "simpler_kernel_mode_prepare_callable");
    api.launch = resolve<decltype(api.launch)>(handle, "simpler_kernel_mode_launch");
    api.finalize = resolve<decltype(api.finalize)>(handle, "finalize_device");
    return api.create_device_context && api.destroy_device_context && api.supported && api.init &&
           api.prepare_callable && api.launch && api.finalize;
}

std::vector<uint8_t> read_binary(const char *path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error(std::string("cannot read runtime binary: ") + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

/// Device ids allocated by CTest resource allocation, parsed from
/// CTEST_RESOURCE_GROUP_<n>_NPUS ("id:<N>,slots:<M>;..."). Mirrors
/// test_comm_lifecycle.cpp::read_ctest_devices().
std::vector<int> read_ctest_devices() {
    std::vector<int> ids;
    const char *count_str = std::getenv("CTEST_RESOURCE_GROUP_COUNT");
    if (count_str == nullptr) return ids;

    int group_count = std::atoi(count_str);
    for (int g = 0; g < group_count; ++g) {
        std::string var = "CTEST_RESOURCE_GROUP_" + std::to_string(g) + "_NPUS";
        const char *val = std::getenv(var.c_str());
        if (val == nullptr) continue;

        std::string s(val);
        size_t pos = 0;
        while ((pos = s.find("id:", pos)) != std::string::npos) {
            pos += 3;
            ids.push_back(std::atoi(s.c_str() + pos));
        }
    }
    return ids;
}

/// A correctly aligned header used only for malformed-input and pre-init tests.
struct AlignedCallableImage {
    alignas(ChipCallable) unsigned char bytes[sizeof(ChipCallable)];

    AlignedCallableImage() { std::memset(bytes, 0, sizeof(bytes)); }
    const void *data() const { return bytes; }
    static size_t size() { return sizeof(ChipCallable); }
};

/// The caller's device and stream, stood up and torn down by the test. This is
/// the state a kernel-mode caller already holds and lends to simpler; nothing
/// under test may reset or finalize any of it.
class BorrowedDevice {
public:
    explicit BorrowedDevice(int device_id) :
        device_id_(device_id) {
        aclError rc = aclInit(nullptr);
        // aclInit is process-wide and another owner in this process may have
        // run it already.
        acl_owned_ = (rc == ACL_SUCCESS);
        if (rc != ACL_SUCCESS && static_cast<int>(rc) != ACL_ERROR_REPEAT_INITIALIZE) return;
        if (aclrtSetDevice(device_id_) != ACL_SUCCESS) return;
        device_set_ = true;
        if (aclrtCreateStream(&stream_) != ACL_SUCCESS) stream_ = nullptr;
    }

    ~BorrowedDevice() {
        if (stream_ != nullptr) aclrtDestroyStream(stream_);
        if (device_set_) aclrtResetDevice(device_id_);
        if (acl_owned_) aclFinalize();
    }

    BorrowedDevice(const BorrowedDevice &) = delete;
    BorrowedDevice &operator=(const BorrowedDevice &) = delete;

    bool ready() const { return device_set_ && stream_ != nullptr; }
    void *stream() const { return stream_; }
    int device_id() const { return device_id_; }

private:
    int device_id_;
    bool acl_owned_{false};
    bool device_set_{false};
    aclrtStream stream_{nullptr};
};

class KernelModeEntryTest : public ::testing::Test {
protected:
    void SetUp() override {
        for (const char *path :
             {PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH,
              PTO_KERNEL_UT_DISPATCHER_PATH}) {
            if (!std::filesystem::exists(path)) {
                GTEST_SKIP() << "onboard runtime artifact not built: " << path
                             << "\n(build the a2a3 onboard tensormap_and_ringbuffer runtime first)";
            }
        }
        auto devices = read_ctest_devices();
        if (devices.empty()) {
            GTEST_SKIP() << "no NPU device allocated; run ctest with --resource-spec-file";
        }
        device_id_ = devices[0];
    }

    int device_id_{-1};
};

// ---------------------------------------------------------------------------
// C ABI surface: all four entries, on a borrowed device and stream.
// ---------------------------------------------------------------------------

TEST_F(KernelModeEntryTest, InitAndFinalizePreserveBorrowedStream) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready()) << "could not stand up the caller's device/stream on device " << device_id_;

    void *handle = dlopen(PTO_HOST_RUNTIME_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << "dlopen failed: " << dlerror();

    KernelModeApi api{};
    ASSERT_TRUE(load_kernel_mode_api(handle, api)) << "kernel-mode entries missing from libhost_runtime.so";

    void *ctx = api.create_device_context();
    ASSERT_NE(ctx, nullptr);

    EXPECT_EQ(api.supported(ctx), 0) << "this backend reports kernel mode unsupported";

    const CallConfig config{};
    const auto aicpu = read_binary(PTO_KERNEL_UT_AICPU_PATH);
    const auto aicore = read_binary(PTO_KERNEL_UT_AICORE_PATH);
    const auto dispatcher = read_binary(PTO_KERNEL_UT_DISPATCHER_PATH);

    EXPECT_EQ(
        api.init(
            ctx, device_id_, aicpu.data(), aicpu.size(), aicore.data(), aicore.size(), dispatcher.data(),
            dispatcher.size(), &config, 1
        ),
        0
    );
    EXPECT_EQ(api.supported(ctx), 0);

    ChipStorageTaskArgs args{};
    EXPECT_EQ(api.launch(ctx, 0, &args, borrowed.stream()), PTO_RUNTIME_ERR_INVALID_STATE)
        << "execution providers are not bound";

    EXPECT_EQ(api.finalize(ctx), 0);
    api.destroy_device_context(ctx);
    dlclose(handle);
    EXPECT_EQ(aclrtSynchronizeStream(borrowed.stream()), ACL_SUCCESS);
}

TEST_F(KernelModeEntryTest, MalformedArgumentsAreRejectedBeforeTheStateCheck) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready());

    void *handle = dlopen(PTO_HOST_RUNTIME_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << "dlopen failed: " << dlerror();
    KernelModeApi api{};
    ASSERT_TRUE(load_kernel_mode_api(handle, api));
    void *ctx = api.create_device_context();
    ASSERT_NE(ctx, nullptr);

    const CallConfig config{};
    const std::vector<uint8_t> aicpu(16, 0);

    // Generation zero is the ABI's invalid generation.
    EXPECT_EQ(
        api.init(ctx, device_id_, aicpu.data(), aicpu.size(), nullptr, 0, nullptr, 0, &config, 0),
        PTO_RUNTIME_ERR_INTERNAL
    );
    // A binary and its size describe one object: present-together or absent-together.
    EXPECT_EQ(api.init(ctx, device_id_, aicpu.data(), 0, nullptr, 0, nullptr, 0, &config, 1), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(
        api.init(ctx, -1, aicpu.data(), aicpu.size(), nullptr, 0, nullptr, 0, &config, 1), PTO_RUNTIME_ERR_INTERNAL
    );

    AlignedCallableImage image;
    EXPECT_EQ(api.prepare_callable(ctx, 0, nullptr, AlignedCallableImage::size()), PTO_RUNTIME_ERR_INTERNAL);
    // Below the size floor the image cannot hold a ChipCallable header.
    EXPECT_EQ(api.prepare_callable(ctx, 0, image.data(), sizeof(ChipCallable) - 1), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(
        api.prepare_callable(ctx, MAX_REGISTERED_CALLABLE_IDS, image.data(), AlignedCallableImage::size()),
        PTO_RUNTIME_ERR_INTERNAL
    );

    ChipStorageTaskArgs args{};
    EXPECT_EQ(api.launch(ctx, 0, &args, nullptr), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(api.launch(ctx, -1, &args, borrowed.stream()), PTO_RUNTIME_ERR_INTERNAL);

    api.destroy_device_context(ctx);
    dlclose(handle);
}

// ---------------------------------------------------------------------------
// ChipWorker surface: the host half of the wiring.
// ---------------------------------------------------------------------------

TEST_F(KernelModeEntryTest, WorkerKernelInitAndClosePreserveBorrowedStream) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready());

    ChipWorker worker;
    const CallConfig config{};
    ASSERT_NO_THROW(worker.kernel_init(
        PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH, PTO_KERNEL_UT_DISPATCHER_PATH,
        device_id_, config, ChipWorker::next_kernel_context_generation()
    ));
    EXPECT_TRUE(worker.initialized());
    EXPECT_EQ(worker.device_id(), device_id_);
    EXPECT_FALSE(worker.kernel_mode_supported());
    ChipStorageTaskArgs args{};
    try {
        worker.kernel_launch(0, &args, borrowed.stream());
        FAIL() << "launch must reject missing execution providers";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find(std::to_string(PTO_RUNTIME_ERR_INVALID_STATE)), std::string::npos);
    }
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_FALSE(worker.initialized());
    EXPECT_EQ(aclrtSynchronizeStream(borrowed.stream()), ACL_SUCCESS);
}

TEST_F(KernelModeEntryTest, WorkerKernelStateMachineRejectsOutOfOrderUse) {
    BorrowedDevice borrowed(device_id_);
    ASSERT_TRUE(borrowed.ready());

    const CallConfig config{};
    AlignedCallableImage image;
    ChipStorageTaskArgs args{};

    // Before any init there is no runtime to reach, so the host half refuses.
    {
        ChipWorker worker;
        EXPECT_THROW(worker.kernel_mode_supported(), std::runtime_error);
        EXPECT_THROW(worker.kernel_prepare_callable(0, image.data(), AlignedCallableImage::size()), std::runtime_error);
        EXPECT_THROW(worker.kernel_launch(0, &args, borrowed.stream()), std::runtime_error);
    }

    // A null stream never reaches the runtime.
    {
        ChipWorker worker;
        EXPECT_THROW(worker.kernel_launch(0, &args, nullptr), std::runtime_error);
    }

    // Generation zero is refused on the host side, before the C ABI would.
    {
        ChipWorker worker;
        EXPECT_THROW(
            worker.kernel_init(
                PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH,
                PTO_KERNEL_UT_DISPATCHER_PATH, device_id_, config, 0
            ),
            std::invalid_argument
        );
    }

    // After finalize the worker is terminal for both identities.
    {
        ChipWorker worker;
        worker.finalize();
        EXPECT_THROW(
            worker.kernel_init(
                PTO_HOST_RUNTIME_LIB_PATH, PTO_KERNEL_UT_AICPU_PATH, PTO_KERNEL_UT_AICORE_PATH,
                PTO_KERNEL_UT_DISPATCHER_PATH, device_id_, config, ChipWorker::next_kernel_context_generation()
            ),
            std::runtime_error
        );
    }
}

TEST_F(KernelModeEntryTest, ContextGenerationsAreNonzeroAndIncreasing) {
    const uint64_t first = ChipWorker::next_kernel_context_generation();
    const uint64_t second = ChipWorker::next_kernel_context_generation();
    EXPECT_NE(first, 0u) << "generation zero is what the C ABI rejects as invalid";
    EXPECT_GT(second, first);
}

}  // namespace
