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
/**
 * AICPU Dispatcher implementation — transient bootstrap-only upload helper.
 *
 * See aicpu_dispatcher.h for architecture. The dispatcher SO exists only
 * to provide a piece of code that runs with sched-thread (HwHiAiUser)
 * permissions for one purpose: write the bundled runtime SO bytes to
 * the main aicpu_scheduler's preinstall path under a content-fingerprint
 * filename. Once Init returns, this SO is no longer referenced — host's
 * subsequent rtsBinaryLoadFromFile calls target the runtime SO file directly.
 */

#include "aicpu_dispatcher.h"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "utils/elf_build_id.h"
#include "so_cache.h"

// dlog wrapper so error paths show up in device log without depending on
// our common/unified_log machinery (this SO is loaded standalone by CANN).
extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);

namespace simpler_dispatcher {
constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelError = 3;

void DispatcherLog(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // DlogRecord is a non-weak extern: if it failed to resolve, this SO
    // would not have dlopen'd in the first place, so an address-vs-nullptr
    // guard here is dead code (and is folded to `true` by most compilers).
    DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[simpler-dispatcher] %s", buf);
}

void AuditCacheSnapshot(
    const std::string &dir, const char *phase, const std::string &target, int fd, size_t len, int same
) {
    const int saved_errno = errno;
    struct statvfs space{};
    const int space_result = statvfs(dir.c_str(), &space);
    const int space_errno = space_result == 0 ? 0 : errno;
    DispatcherLog(
        "SO_CACHE_AUDIT_2651 phase=%s pid=%d root=%s space_rc=%d space_errno=%d blocks=%llu free=%llu "
        "available=%llu frsize=%llu inodes=%llu ifree=%llu",
        phase, static_cast<int>(getpid()), dir.c_str(), space_result, space_errno,
        static_cast<unsigned long long>(space.f_blocks), static_cast<unsigned long long>(space.f_bfree),
        static_cast<unsigned long long>(space.f_bavail), static_cast<unsigned long long>(space.f_frsize),
        static_cast<unsigned long long>(space.f_files), static_cast<unsigned long long>(space.f_ffree)
    );
    uint64_t scanned = 0, originals = 0, trials = 0, temps = 0, unknown = 0, special = 0;
    uint64_t bytes = 0, allocated = 0, original_allocated = 0, trial_allocated = 0;
    uint64_t errors = 0, detailed = 0, other_originals = 0;
    struct stat directory{};
    const int directory_result = lstat(dir.c_str(), &directory);
    if (directory_result != 0 || !S_ISDIR(directory.st_mode)) ++errors;
    DispatcherLog(
        "SO_CACHE_AUDIT_2651 phase=%s pid=%d directory_rc=%d directory_dev=%llu directory_inode=%llu", phase,
        static_cast<int>(getpid()), directory_result, static_cast<unsigned long long>(directory.st_dev),
        static_cast<unsigned long long>(directory.st_ino)
    );
    DIR *entries = opendir(dir.c_str());
    if (entries == nullptr) ++errors;
    else {
        while (true) {
            errno = 0;
            auto *entry = readdir(entries);
            if (entry == nullptr) {
                if (errno != 0) ++errors;
                break;
            }
            const std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            ++scanned;
            if (name.rfind("simpler_inner_", 0) != 0) continue;
            struct stat st{};
            if (lstat((dir + "/" + name).c_str(), &st) != 0) {
                ++errors;
                continue;
            }
            if (detailed < 4096) {
                DispatcherLog(
                    "SO_CACHE_AUDIT_2651 phase=%s pid=%d entry=%s dev=%llu inode=%llu mode=%o nlink=%llu "
                    "size=%llu allocated=%llu is_target=%d",
                    phase, static_cast<int>(getpid()), name.c_str(), static_cast<unsigned long long>(st.st_dev),
                    static_cast<unsigned long long>(st.st_ino), static_cast<unsigned>(st.st_mode),
                    static_cast<unsigned long long>(st.st_nlink), static_cast<unsigned long long>(st.st_size),
                    static_cast<unsigned long long>(st.st_blocks) * 512, dir + "/" + name == target
                );
                ++detailed;
            }
            if (!S_ISREG(st.st_mode)) {
                ++special;
                continue;
            }
            const bool original = IsManagedSo(name.c_str());
            const bool trial =
                name.rfind("simpler_inner_v2_", 0) == 0 && IsManagedSo(("simpler_inner_" + name.substr(17)).c_str());
            if (original) {
                ++originals;
                if (dir + "/" + name != target) ++other_originals;
            } else if (trial) ++trials;
            else ++unknown;
            if ((original || trial) && name.find(".tmp.") != std::string::npos) ++temps;
            const uint64_t used = static_cast<uint64_t>(st.st_blocks) * 512;
            bytes += st.st_size;
            allocated += used;
            if (original) original_allocated += used;
            if (trial) trial_allocated += used;
        }
        if (closedir(entries) != 0) ++errors;
    }
    DispatcherLog(
        "SO_CACHE_AUDIT_2651 phase=%s pid=%d scanned=%llu original=%llu v2=%llu temps=%llu unknown=%llu "
        "special=%llu size=%llu allocated=%llu original_allocated=%llu v2_allocated=%llu errors=%llu complete=%d",
        phase, static_cast<int>(getpid()), static_cast<unsigned long long>(scanned),
        static_cast<unsigned long long>(originals), static_cast<unsigned long long>(trials),
        static_cast<unsigned long long>(temps), static_cast<unsigned long long>(unknown),
        static_cast<unsigned long long>(special), static_cast<unsigned long long>(bytes),
        static_cast<unsigned long long>(allocated), static_cast<unsigned long long>(original_allocated),
        static_cast<unsigned long long>(trial_allocated), static_cast<unsigned long long>(errors), errors == 0
    );
    const bool complete = errors == 0 && space_result == 0;
    DispatcherLog(
        "SO_CACHE_AUDIT_2651 phase=%s pid=%d detailed=%llu manifest_complete=%d other_original=%llu "
        "zero_original=%d zero_all_project=%d",
        phase, static_cast<int>(getpid()), static_cast<unsigned long long>(detailed),
        complete && detailed == originals + trials + unknown + special,
        static_cast<unsigned long long>(other_originals), complete && originals == 0 && special == 0 && unknown == 0,
        complete && originals + trials + unknown + special == 0
    );
    struct stat held{}, current{};
    const int held_result = fd < 0 ? -1 : fstat(fd, &held);
    const int path_result = lstat(target.c_str(), &current);
    DispatcherLog(
        "SO_CACHE_AUDIT_2651 phase=%s pid=%d target=%s fd=%d fstat_rc=%d inode=%llu nlink=%llu size=%llu "
        "expected=%llu same=%d path_rc=%d path_inode=%llu path_matches=%d",
        phase, static_cast<int>(getpid()), target.c_str(), fd, held_result,
        static_cast<unsigned long long>(held.st_ino), static_cast<unsigned long long>(held.st_nlink),
        static_cast<unsigned long long>(held.st_size), static_cast<unsigned long long>(len), same, path_result,
        static_cast<unsigned long long>(current.st_ino),
        held_result == 0 && path_result == 0 && held.st_dev == current.st_dev && held.st_ino == current.st_ino
    );
    errno = saved_errno;
}

void AuditCacheRemoved(const std::string &path, int fd, int result) {
    const int saved_errno = errno;
    struct stat held{};
    int status = fstat(fd, &held);
    DispatcherLog(
        "SO_CACHE_AUDIT_2651 phase=unlink pid=%d target=%s rc=%d errno=%d fstat_rc=%d inode=%llu nlink=%llu "
        "size=%llu allocated=%llu",
        static_cast<int>(getpid()), path.c_str(), result, result == 0 ? 0 : saved_errno, status,
        static_cast<unsigned long long>(held.st_ino), static_cast<unsigned long long>(held.st_nlink),
        static_cast<unsigned long long>(held.st_size), static_cast<unsigned long long>(held.st_blocks) * 512
    );
    errno = saved_errno;
}

void AuditCacheGc(uint64_t matched, uint64_t removed, uint64_t busy, uint64_t errors) {
    const int saved_errno = errno;
    DispatcherLog(
        "SO_CACHE_AUDIT_2651 phase=gc pid=%d matched=%llu removed=%llu busy=%llu errors=%llu",
        static_cast<int>(getpid()), static_cast<unsigned long long>(matched), static_cast<unsigned long long>(removed),
        static_cast<unsigned long long>(busy), static_cast<unsigned long long>(errors)
    );
    errno = saved_errno;
}
}  // namespace simpler_dispatcher

// Bootstrap-time DeviceArgs view. Layout shared with host's BootstrapDispatcher.
// libaicpu_extend_kernels reads aicpu_so_bin/len/deviceId; we additionally read
// inner_so_bin/len (an extra qword pair past deviceId).
struct KernelArgs {
    uint64_t unused[5] = {0};
    void *device_args{nullptr};
    void *runtime_args{nullptr};
    uint64_t regs{0};
};
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpu_so_bin{0};  // 96  — dispatcher bytes (libaicpu_extend_kernels)
    uint64_t aicpu_so_len{0};  // 104
    uint64_t device_id{0};     // 112
    uint64_t inner_so_bin{0};  // 120 — runtime SO bytes (dispatcher)
    uint64_t inner_so_len{0};  // 128
};
static_assert(offsetof(KernelArgs, device_args) == 40, "KernelArgs::device_args offset drift");
static_assert(offsetof(DeviceArgs, aicpu_so_bin) == 96, "DeviceArgs::aicpu_so_bin offset drift");
static_assert(offsetof(DeviceArgs, aicpu_so_len) == 104, "DeviceArgs::aicpu_so_len offset drift");
static_assert(offsetof(DeviceArgs, device_id) == 112, "DeviceArgs::device_id offset drift");
static_assert(offsetof(DeviceArgs, inner_so_bin) == 120, "DeviceArgs::inner_so_bin offset drift");
static_assert(offsetof(DeviceArgs, inner_so_len) == 128, "DeviceArgs::inner_so_len offset drift");

namespace simpler_dispatcher {

// ELF Build-ID-derived 64-bit fingerprint (linker SHA-1 truncated to 8
// bytes by `-Wl,--build-id`). Falls back to full-buffer FNV-1a if the SO
// was somehow linked without a build-id note. Host's
// load_aicpu_op.cpp::FingerprintBytes calls the same helper, so both sides
// produce identical fingerprints with no other channel of communication.
//
// The earlier "FNV-1a over the first 64 bytes XOR len" scheme collided in
// practice on same-toolchain runtime SOs whose ELF headers + size matched
// — wrong-code risk on the multi-runtime path. Build-IDs are strong by
// linker contract: identical Build-IDs imply byte-identical SOs.
uint64_t Fingerprint(const char *data, uint64_t len) {
    return simpler::common::utils::elf_build_id_64(data, static_cast<std::size_t>(len));
}

// Preinstall path — HwHiAiUser owns this dir, the sched thread can write here.
// device-side /tmp is mounted read-only / restricted in CANN 9.0.
std::string MakeInnerSoPath(uint64_t fp, uint64_t device_id) {
    char buf[256];
    snprintf(
        buf, sizeof(buf), "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_%016lx_%lu.so", fp, device_id
    );
    return buf;
}

// Atomic write: write to a per-process temp path, then rename onto the target.
// Several CI workers may bootstrap on different devices simultaneously and all
// land at the same fingerprinted target path; without atomic rename a reader
// (a sibling aicpu_scheduler's dlopen during its rtsBinaryLoadFromFile call) can observe a
// truncated/partially-written file and fail with 507018 or 507046.
//
// Same fingerprint → same content, so whichever rename wins yields identical
// bytes; existing dlopen handles in any aicpu_scheduler stay bound to their
// captured inode and are unaffected by later renames. We don't fast-path on
// the file already existing — a stale corrupt file from a pre-fix run could
// match the fingerprint by chance, and the atomic rename overwrites cheaply.
bool WriteBytes(const std::string &path, const char *data, uint64_t len) {
    char tmp_path[320];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", path.c_str(), static_cast<int>(getpid()));
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            DispatcherLog("open %s for write failed: %s", tmp_path, strerror(errno));
            return false;
        }
        f.write(data, static_cast<std::streamsize>(len));
        f.close();
        bool good = f.good();
        if (!good) {
            DispatcherLog("write %s failed", tmp_path);
            unlink(tmp_path);
            return false;
        }
    }
    if (chmod(tmp_path, 0755) != 0) {
        unlink(tmp_path);
        return false;
    }
    if (rename(tmp_path, path.c_str()) != 0) {
        DispatcherLog("rename %s -> %s failed: %s", tmp_path, path.c_str(), strerror(errno));
        unlink(tmp_path);
        return false;
    }
    return true;
}

}  // namespace simpler_dispatcher

// =============================================================================
// C-style exported entry points dlsym'd by libaicpu_extend_kernels.
// =============================================================================

extern "C" {

// Stubs — libaicpu_extend_kernels::SetTileFwkKernelMap dlsym's all three at
// load time; absence makes the whole SO unmappable. We only reach Init in
// practice, but return 0 (success) here to mirror the happy-path return of
// the old AICPU kernel stubs we replaced. If a future CANN version begins
// invoking Static as a warm-up probe, returning failure would be a silent
// regression versus the prior behavior.
__attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *args) {
    (void)args;
    simpler_dispatcher::DispatcherLog("Static: stub (not expected to be called)");
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServer(void *args) {
    (void)args;
    simpler_dispatcher::DispatcherLog("Server: stub (dispatcher is upload-only, not expected to be called)");
    return 0;
}

// Init: write the bundled runtime SO bytes to a fingerprint-named file under
// the main scheduler's preinstall path, return. Once this returns, host's
// host's subsequent rtsBinaryLoadFromFile can resolve the runtime SO directly — this dispatcher SO
// never gets referenced again.
__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServerInit(void *args) {
    if (args == nullptr) {
        simpler_dispatcher::DispatcherLog("Init: args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<DeviceArgs *>(k->device_args);
    if (d == nullptr) {
        simpler_dispatcher::DispatcherLog("Init: device_args==nullptr");
        return 1;
    }
    if (d->inner_so_bin == 0 || d->inner_so_len == 0) {
        simpler_dispatcher::DispatcherLog(
            "Init: empty inner SO bundle (bin=%lx len=%lu)", d->inner_so_bin, d->inner_so_len
        );
        return 1;
    }
    const char *inner_bytes = reinterpret_cast<const char *>(d->inner_so_bin);
    uint64_t fp = simpler_dispatcher::Fingerprint(inner_bytes, d->inner_so_len);
    std::string path = simpler_dispatcher::MakeInnerSoPath(fp, d->device_id);
    size_t start = path.rfind('/') + 1;
    if (!simpler_dispatcher::StageManagedSo(
            path.substr(0, start - 1), path.substr(start), inner_bytes, d->inner_so_len
        )) {
        simpler_dispatcher::DispatcherLog("Init: stage %s failed: %s", path.c_str(), strerror(errno));
        return 1;
    }
    simpler_dispatcher::DispatcherLog("Init: wrote %s (%lu bytes)", path.c_str(), d->inner_so_len);
    return 0;
}

}  // extern "C"
