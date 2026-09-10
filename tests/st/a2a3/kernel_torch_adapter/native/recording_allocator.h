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

#pragma once

#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <map>
#include <mutex>
#include <stdexcept>

namespace kernel_adapter_test {
using c10_npu::MempoolId_t;
using namespace c10_npu::NPUCachingAllocator;

// Test-only interposition at the SDK virtual call boundary; every operation
// still reaches the real allocator, including recordStream itself.
class RecordingAllocator final : public NPUAllocator {
public:
    explicit RecordingAllocator(NPUAllocator *delegate) :
        delegate_(delegate) {}
    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *expected = delegate_;
        if (!allocator.compare_exchange_strong(expected, this))
            throw std::runtime_error("allocator probe is already active or allocator changed");
        counts_.clear();
    }
    void stop() {
        auto *expected = static_cast<NPUAllocator *>(this);
        if (!allocator.compare_exchange_strong(expected, delegate_))
            throw std::runtime_error("allocator probe is not active");
    }
    size_t count(uintptr_t address) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = counts_.find(address);
        return it == counts_.end() ? 0 : it->second;
    }
    void recordStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++counts_[reinterpret_cast<uintptr_t>(ptr.get())];
        }
        delegate_->recordStream(ptr, stream);
    }
    c10::DataPtr allocate(size_t bytes) override { return delegate_->allocate(bytes); }
    c10::DeleterFnPtr raw_deleter() const override { return delegate_->raw_deleter(); }
    void copy_data(void *dst, const void *src, size_t bytes) const override { delegate_->copy_data(dst, src, bytes); }
    c10::DataPtr allocate_with_aligned(size_t size, size_t aligned) const override {
        return delegate_->allocate_with_aligned(size, aligned);
    }
    void *raw_alloc(size_t bytes) override { return delegate_->raw_alloc(bytes); }
    void *raw_alloc_with_stream(size_t bytes, aclrtStream stream) override {
        return delegate_->raw_alloc_with_stream(bytes, stream);
    }
    void raw_delete(void *ptr) override { delegate_->raw_delete(ptr); }
    void init(int count) override { delegate_->init(count); }
    bool initialized() override { return delegate_->initialized(); }
    void setMemoryFraction(double fraction, int device) override { delegate_->setMemoryFraction(fraction, device); }
    void emptyCacheImpl(bool check, bool physical) override { delegate_->emptyCacheImpl(check, physical); }
    void emptyCache(bool check) override { delegate_->emptyCache(check); }
    void emptyVirtAddrCache(bool check) override { delegate_->emptyVirtAddrCache(check); }
    void cacheInfo(int device, size_t *free, size_t *largest) override { delegate_->cacheInfo(device, free, largest); }
    void *getBaseAllocation(void *ptr, size_t *size) override { return delegate_->getBaseAllocation(ptr, size); }
    void eraseStream(const c10::DataPtr &ptr, c10_npu::NPUStream stream) override {
        delegate_->eraseStream(ptr, stream);
    }
    DeviceStats getDeviceStats(int device) override { return delegate_->getDeviceStats(device); }
    void resetAccumulatedStats(int device) override { delegate_->resetAccumulatedStats(device); }
    void resetPeakStats(int device) override { delegate_->resetPeakStats(device); }
    SnapshotInfo snapshot() override { return delegate_->snapshot(); }
    void
    beginAllocateToPool(c10::DeviceIndex device, MempoolId_t pool, std::function<bool(aclrtStream)> filter) override {
        delegate_->beginAllocateToPool(device, pool, std::move(filter));
    }
    void endAllocateToPool(c10::DeviceIndex device, MempoolId_t pool) override {
        delegate_->endAllocateToPool(device, pool);
    }
    void releasePool(c10::DeviceIndex device, MempoolId_t pool) override { delegate_->releasePool(device, pool); }
    void FreeDeviceCachedMemory(int device) override { delegate_->FreeDeviceCachedMemory(device); }
    std::string name() override { return delegate_->name(); }
    bool checkPoolLiveAllocations(
        c10::DeviceIndex device, MempoolId_t pool, const std::unordered_set<void *> &live
    ) override {
        return delegate_->checkPoolLiveAllocations(device, pool, live);
    }
    ShareableHandle shareIpcHandle(void *ptr) override { return delegate_->shareIpcHandle(ptr); }
    std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
        return delegate_->getIpcDevPtr(std::move(handle));
    }
    bool isHistoryEnabled() override { return delegate_->isHistoryEnabled(); }
    void recordHistory(bool enabled, CreateContextFn recorder, size_t entries, RecordContext when) override {
        delegate_->recordHistory(enabled, recorder, entries, when);
    }
    void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
        delegate_->attachOutOfMemoryObserver(std::move(observer));
    }
    bool checkUceInMemPool(int device) override { return delegate_->checkUceInMemPool(device); }
    bool checkBlockIsSafe(const c10::DataPtr &ptr) override { return delegate_->checkBlockIsSafe(ptr); }
    void markAllBlockUnsafe(int device) override { delegate_->markAllBlockUnsafe(device); }
    void updateBlockToSafe(const c10::DataPtr &ptr) override { delegate_->updateBlockToSafe(ptr); }
    void cleanEvent() override { delegate_->cleanEvent(); }
    void buildServerMemMapForHccl(int device, std::shared_ptr<c10d_npu::HCCLComm> comm) override {
        delegate_->buildServerMemMapForHccl(device, std::move(comm));
    }
    std::shared_ptr<AllocatorState> getCheckpointState(c10::DeviceIndex device, MempoolId_t pool) override {
        return delegate_->getCheckpointState(device, pool);
    }
    CheckpointDelta setCheckpointPoolState(c10::DeviceIndex device, std::shared_ptr<AllocatorState> state) override {
        return delegate_->setCheckpointPoolState(device, std::move(state));
    }

private:
    NPUAllocator *delegate_;
    std::mutex mutex_;
    std::map<uintptr_t, size_t> counts_;
};

inline RecordingAllocator &recording_allocator() {
    // SDK release threads may still hold a loaded proxy pointer after stop.
    // This one test-process object remains valid until process termination.
    static auto *probe = new RecordingAllocator(get());
    return *probe;
}
}  // namespace kernel_adapter_test
