/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <mutex>

/**
 * Host-side submission gate for one kernel context.
 *
 * The gate serializes invocation borrowing, enqueue completion and close. It
 * does not own streams, events, callable records or device allocations; those
 * remain in their existing providers. The caller stream is borrowed only while
 * the context is live and is never retained or destroyed by this object.
 */
class KernelSubmissionState {
public:
    KernelSubmissionState() = default;
    KernelSubmissionState(const KernelSubmissionState &) = delete;
    KernelSubmissionState &operator=(const KernelSubmissionState &) = delete;

    std::unique_lock<std::mutex> lock() { return std::unique_lock<std::mutex>(mutex_); }

    uintptr_t previous_caller() const noexcept { return previous_caller_; }
    void *previous_caller_stream() const noexcept { return previous_caller_stream_; }
    bool has_tail() const noexcept { return has_tail_; }
    bool closing() const noexcept { return closing_; }

    void commit_tail(uintptr_t caller, void *caller_stream) noexcept {
        previous_caller_ = caller;
        previous_caller_stream_ = caller_stream;
        has_tail_ = caller != 0;
    }

    // Compatibility helper for state-only unit tests; production submissions
    // use the overload carrying the borrowed stream pointer.
    void commit_tail(uintptr_t caller) noexcept { commit_tail(caller, nullptr); }

    void clear_tail() noexcept {
        previous_caller_ = 0;
        previous_caller_stream_ = nullptr;
        has_tail_ = false;
    }

    void begin_close() noexcept { closing_ = true; }

private:
    mutable std::mutex mutex_;
    uintptr_t previous_caller_{0};
    void *previous_caller_stream_{nullptr};
    bool has_tail_{false};
    bool closing_{false};
};
