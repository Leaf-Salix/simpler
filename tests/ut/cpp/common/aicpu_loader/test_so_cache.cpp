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
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "aicpu_loader/device/so_cache.h"

extern "C" void DlogRecord(int, int, const char *, ...) {}

namespace {
namespace fs = std::filesystem;
using simpler_dispatcher::StageManagedSo;

std::string Name(int id) {
    char name[80];
    snprintf(name, sizeof(name), "simpler_inner_v2_%016x_0.so", id);
    return name;
}
std::string Read(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
}
void Write(const std::string &path, const std::string &data) { std::ofstream(path, std::ios::binary) << data; }

// All staging happens in children so process exit, rather than a test-only
// release API, drops the production leases.
class CacheOwner {
public:
    pid_t pid = -1;
    int command = -1;
    bool ready = false;
    explicit CacheOwner(
        const std::function<bool()> &stage, const std::function<bool()> &finish = [] {
            return true;
        }
    ) {
        int started[2], release[2];
        if (pipe(started) != 0 || pipe(release) != 0) return;
        pid = fork();
        if (pid == 0) {
            close(started[0]);
            close(release[1]);
            char result = stage() ? 1 : 0;
            if (write(started[1], &result, 1) != 1) _exit(2);
            if (read(release[0], &result, 1) != 1) _exit(3);
            _exit(finish() ? 0 : 4);
        }
        close(started[1]);
        close(release[0]);
        command = release[1];
        char result = 0;
        ready = pid > 0 && read(started[0], &result, 1) == 1 && result == 1;
        close(started[0]);
    }
    int Stop(int sig = 0) {
        if (pid <= 0) return -1;
        if (sig) kill(pid, sig);
        else {
            char value = 1;
            (void)write(command, &value, 1);
        }
        close(command);
        command = -1;
        int status = 0;
        waitpid(pid, &status, 0);
        pid = -1;
        return status;
    }
    ~CacheOwner() {
        if (pid > 0) Stop(SIGKILL);
    }
};

class SoCache : public testing::Test {
protected:
    std::string dir;
    void SetUp() override {
        char path[] = "/tmp/simpler-so-cache-XXXXXX";
        ASSERT_NE(mkdtemp(path), nullptr);
        dir = path;
    }
    void TearDown() override { fs::remove_all(dir); }
    bool Stage(int id, const std::string &bytes = "runtime") {
        return StageManagedSo(dir, Name(id), bytes.data(), bytes.size());
    }
    void Once(const std::function<bool()> &fn) {
        CacheOwner child(fn);
        ASSERT_TRUE(child.ready);
        ASSERT_EQ(child.Stop(), 0);
    }
};

TEST_F(SoCache, ReclaimsHistoricalBuildsAndTempsButPreservesForeignFiles) {
    Write(dir + "/simpler_inner_0000000000000001_0.so", "legacy");
    Write(dir + "/libother.so", "foreign");
    Write(dir + "/" + Name(99) + ".tmp.123", "partial");
    ASSERT_EQ(symlink("libother.so", (dir + "/" + Name(98)).c_str()), 0);
    ASSERT_EQ(mkfifo((dir + "/" + Name(97)).c_str(), 0600), 0);
    for (int id = 1; id <= 12; ++id) {
        Once([&] {
            return Stage(id);
        });
        if (id > 1) EXPECT_FALSE(fs::exists(dir + "/" + Name(id - 1)));
        EXPECT_EQ(Read(dir + "/" + Name(id)), "runtime");
    }
    EXPECT_FALSE(fs::exists(dir + "/" + Name(99) + ".tmp.123"));
    EXPECT_EQ(Read(dir + "/simpler_inner_0000000000000001_0.so"), "legacy");
    EXPECT_EQ(Read(dir + "/libother.so"), "foreign");
    EXPECT_TRUE(fs::is_symlink(dir + "/" + Name(98)));
    EXPECT_TRUE(fs::is_fifo(dir + "/" + Name(97)));
}

TEST_F(SoCache, ProtectsPendingLoadAndSharedReadersUntilLastProcessExits) {
    const std::string path = dir + "/" + Name(1);
    CacheOwner first(
        [&] {
            return Stage(1);
        },
        [&] {
            return Read(path) == "runtime";
        }
    );
    ASSERT_TRUE(first.ready);
    struct stat before{};
    ASSERT_EQ(stat(path.c_str(), &before), 0);
    CacheOwner second([&] {
        return Stage(1);
    });
    ASSERT_TRUE(second.ready);
    struct stat after{};
    ASSERT_EQ(stat(path.c_str(), &after), 0);
    EXPECT_EQ(before.st_ino, after.st_ino);  // Cache hit needs no second SO allocation.
    Once([&] {
        return Stage(2);
    });
    ASSERT_EQ(first.Stop(), 0);  // Delayed first read still resolves the published path.
    Once([&] {
        return Stage(3);
    });
    EXPECT_TRUE(fs::exists(path));
    int status = second.Stop(SIGKILL);
    ASSERT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGKILL);
    Once([&] {
        return Stage(4);
    });
    EXPECT_FALSE(fs::exists(path));
}

TEST_F(SoCache, RepairsUnleasedCorruptionButNeverReplacesALeasedInode) {
    const std::string path = dir + "/" + Name(1);
    for (const auto &bad : {"runtim", "wrong!!", "runtime-extra"}) {
        Write(path, bad);
        Once([&] {
            return Stage(1);
        });
        EXPECT_EQ(Read(path), "runtime");
    }
    CacheOwner owner([&] {
        return Stage(1);
    });
    ASSERT_TRUE(owner.ready);
    Once([&] {
        return !Stage(1, "different");
    });
    EXPECT_EQ(Read(path), "runtime");
}

TEST_F(SoCache, FailedWritePreservesTargetAndRemovesPartialUpload) {
    const std::string path = dir + "/" + Name(1);
    Write(path, "old");
    Once([&] {
        signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit{4096, 4096};
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) return false;
        return !Stage(1, std::string(1024 * 1024, 'X'));
    });
    EXPECT_EQ(Read(path), "old");
    for (const auto &entry : fs::directory_iterator(dir))
        EXPECT_EQ(entry.path().string().find(".tmp."), std::string::npos);
    Once([&] {
        return Stage(1);
    });
    EXPECT_EQ(Read(path), "runtime");
}

TEST_F(SoCache, ConcurrentThreadsKeepAllActiveLeasesAndReuseDescriptors) {
    Once([&] {
        struct rlimit limit{64, 64};
        if (setrlimit(RLIMIT_NOFILE, &limit) != 0) return false;
        bool good[2] = {true, true};
        std::thread threads[2];
        for (int i = 0; i < 2; ++i) {
            threads[i] = std::thread([&, i] {
                for (int repeat = 0; repeat < 50; ++repeat) {
                    if (!Stage(i + 1) || !Stage(3)) good[i] = false;
                }
            });
        }
        for (auto &thread : threads)
            thread.join();
        return good[0] && good[1] && Read(dir + "/" + Name(1)) == "runtime" && Read(dir + "/" + Name(2)) == "runtime" &&
               Read(dir + "/" + Name(3)) == "runtime";
    });
    Once([&] {
        return Stage(4);
    });
    for (int id = 1; id <= 3; ++id)
        EXPECT_FALSE(fs::exists(dir + "/" + Name(id)));
}

}  // namespace
