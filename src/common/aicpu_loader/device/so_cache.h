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

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>

namespace simpler_dispatcher {

struct CacheFd {
    int fd;
    explicit CacheFd(int value) :
        fd(value) {}
    ~CacheFd() {
        if (fd >= 0) close(fd);
    }
    CacheFd(const CacheFd &) = delete;
    CacheFd &operator=(const CacheFd &) = delete;
};

inline bool IsManagedSo(const char *name) {
    char fp[17] = {}, device[21] = {};
    int end = 0;
    sscanf(name, "simpler_inner_%16[0123456789abcdef]_%20[0123456789].so%n", fp, device, &end);
    if (strlen(fp) != 16 || end == 0) return false;
    if (name[end] == '\0') return true;
    if (strncmp(name + end, ".tmp.", 5) != 0) return false;
    const char *pid = name + end + 5;
    return *pid != '\0' && strspn(pid, "0123456789") == strlen(pid);
}

inline int OpenCacheFile(const std::string &path, int flags) {
    int fd = open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0600);
    if (fd < 0) return -1;
    struct stat st{};
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) return fd;
    close(fd);
    errno = EINVAL;
    return -1;
}

inline int SameBytes(int fd, const char *data, size_t len) {
    struct stat st{};
    if (fstat(fd, &st) != 0) return -1;
    if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) != len) return 0;
    char buf[16384];
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = pread(fd, buf, std::min(sizeof(buf), len - offset), offset);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return -1;
        if (n == 0 || memcmp(buf, data + offset, n) != 0) return 0;
        offset += n;
    }
    return 1;
}

void AuditCacheSnapshot(
    const std::string &dir, const char *phase, const std::string &target, int fd, size_t len, int same
);
void AuditCacheRemoved(const std::string &path, int fd, int result);
void AuditCacheGc(uint64_t matched, uint64_t removed, uint64_t busy, uint64_t errors);

inline void ReclaimUnusedSos(const std::string &dir, const std::string &target) {
    DIR *entries = opendir(dir.c_str());
    if (entries == nullptr) return;
    uint64_t matched = 0, removed = 0, busy = 0, errors = 0;
    while (auto *entry = readdir(entries)) {
        if (!IsManagedSo(entry->d_name)) continue;
        std::string path = dir + "/" + entry->d_name;
        if (path == target) continue;
        ++matched;
        CacheFd candidate(OpenCacheFile(path, O_RDONLY));
        if (candidate.fd < 0) {
            ++errors;
        } else if (flock(candidate.fd, LOCK_EX | LOCK_NB) != 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) ++busy;
            else ++errors;
        } else {
            int result = unlink(path.c_str());
            if (result == 0) ++removed;
            else ++errors;
            if (removed + errors <= 8) AuditCacheRemoved(path, candidate.fd, result);
        }
    }
    closedir(entries);
    AuditCacheGc(matched, removed, busy, errors);
}

bool WriteBytes(const std::string &path, const char *data, uint64_t len);

// The global lock serializes publication, reclamation and the lease table.
// Each call opens a separate lock FD: flock on one shared FD does not serialize
// threads. Shared SO locks live until the scheduler process exits (even _exit
// or SIGKILL), protecting both pending first dlopen and later registrations.
inline bool StageManagedSo(const std::string &dir, const std::string &name, const char *data, size_t len) {
    CacheFd guard(OpenCacheFile(dir + "/.simpler_so_cache.lock", O_CREAT | O_RDWR));
    if (guard.fd < 0 || flock(guard.fd, LOCK_EX) != 0) return false;
    const std::string path = dir + "/" + name;
    CacheFd target(OpenCacheFile(path, O_RDONLY));
    int same = 0;
    if (target.fd >= 0) {
        if (flock(target.fd, LOCK_SH | LOCK_NB) != 0) return false;
        same = SameBytes(target.fd, data, len);
        if (same < 0) return false;
        // Replacing a leased inode would leave its users protecting the wrong
        // pathname. Fail closed on corruption/collision while it is in use.
        if (same == 0 && flock(target.fd, LOCK_EX | LOCK_NB) != 0) return false;
    } else if (errno != ENOENT) {
        return false;
    }
    AuditCacheSnapshot(dir, "before_gc", path, target.fd, len, same);
    ReclaimUnusedSos(dir, path);
    AuditCacheSnapshot(dir, "after_gc_before_publish", path, target.fd, len, same);
    if (same == 0) {
        {
            CacheFd output(OpenCacheFile(path + ".tmp." + std::to_string(getpid()), O_CREAT | O_EXCL | O_WRONLY));
            if (output.fd < 0) return false;
        }
        if (!WriteBytes(path, data, len)) return false;
        if (target.fd >= 0) close(target.fd);
        target.fd = OpenCacheFile(path, O_RDONLY);
        if (target.fd < 0 || flock(target.fd, LOCK_SH | LOCK_NB) != 0) return false;
    }
    // Deliberately retain FDs across DSO unloading; the OS releases leases at
    // process exit. A repeated bootstrap retains just one descriptor per SO.
    static std::map<std::string, int> leases;
    auto found = leases.find(path);
    if (found != leases.end()) close(found->second);
    leases[path] = target.fd;
    AuditCacheSnapshot(dir, "leased", path, target.fd, len, same);
    target.fd = -1;
    return true;
}

}  // namespace simpler_dispatcher
