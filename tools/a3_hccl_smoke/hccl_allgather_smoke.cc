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
#include <hccl/hccl.h>
#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool check_acl(aclError ret, const char *expr, const char *file, int line) {
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "[ACL ERROR] " << file << ":" << line << " " << expr << " -> " << ret << "\n";
    return false;
}

bool check_hccl(HcclResult ret, const char *expr, const char *file, int line) {
    if (ret == HCCL_SUCCESS) {
        return true;
    }
    std::cerr << "[HCCL ERROR] " << file << ":" << line << " " << expr << " -> " << ret << "\n";
    return false;
}

#define CHECK_ACL(expr) check_acl((expr), #expr, __FILE__, __LINE__)
#define CHECK_HCCL(expr) check_hccl((expr), #expr, __FILE__, __LINE__)

int env_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::stoi(value);
}

int mpi_local_rank() {
    int fallback = -1;
    fallback = env_int("OMPI_COMM_WORLD_LOCAL_RANK", fallback);
    fallback = env_int("MPI_LOCALRANKID", fallback);
    fallback = env_int("MV2_COMM_WORLD_LOCAL_RANK", fallback);
    if (fallback >= 0) {
        return fallback;
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    return local_rank;
}

void log_pre_mpi(const std::string &message) { std::cerr << "[pre-mpi] " << message << std::endl; }

void log_rank(int rank, const std::string &message) { std::cerr << "[rank " << rank << "] " << message << std::endl; }

}  // namespace

int main(int argc, char **argv) {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    log_pre_mpi("MPI_Init begin");
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <ranktable.json> [count_per_rank]\n";
        }
        MPI_Finalize();
        return 2;
    }

    const std::string ranktable_path = argv[1];
    const uint64_t count = argc >= 3 ? std::stoull(argv[2]) : 16;
    const int device_id = mpi_local_rank();

    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    void *send_buf = nullptr;
    void *recv_buf = nullptr;
    bool acl_inited = false;
    bool device_set = false;
    int rc = 1;

    do {
        log_rank(
            rank, "MPI_Init OK world_size=" + std::to_string(world_size) + " local_device=" + std::to_string(device_id)
        );
        log_rank(rank, "aclInit begin");
        if (!CHECK_ACL(aclInit(nullptr))) {
            break;
        }
        acl_inited = true;
        log_rank(rank, "aclInit OK");

        log_rank(rank, "aclrtSetDevice begin device=" + std::to_string(device_id));
        if (!CHECK_ACL(aclrtSetDevice(device_id))) {
            break;
        }
        device_set = true;
        log_rank(rank, "aclrtSetDevice OK");

        log_rank(rank, "HcclCommInitClusterInfo begin ranktable=" + ranktable_path);
        if (!CHECK_HCCL(HcclCommInitClusterInfo(ranktable_path.c_str(), static_cast<uint32_t>(rank), &comm))) {
            break;
        }
        log_rank(rank, "HcclCommInitClusterInfo OK");

        log_rank(rank, "aclrtCreateStream begin");
        if (!CHECK_ACL(aclrtCreateStream(&stream))) {
            break;
        }
        log_rank(rank, "aclrtCreateStream OK");

        const size_t send_bytes = count * sizeof(int32_t);
        const size_t recv_bytes = count * static_cast<uint64_t>(world_size) * sizeof(int32_t);
        log_rank(
            rank,
            "aclrtMalloc begin send_bytes=" + std::to_string(send_bytes) + " recv_bytes=" + std::to_string(recv_bytes)
        );
        if (!CHECK_ACL(aclrtMalloc(&send_buf, send_bytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
            break;
        }
        if (!CHECK_ACL(aclrtMalloc(&recv_buf, recv_bytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
            break;
        }
        log_rank(rank, "aclrtMalloc OK");

        std::vector<int32_t> host_send(count, static_cast<int32_t>(rank));
        std::vector<int32_t> host_recv(count * static_cast<uint64_t>(world_size), -1);
        log_rank(rank, "input copy begin");
        if (!CHECK_ACL(aclrtMemcpy(send_buf, send_bytes, host_send.data(), send_bytes, ACL_MEMCPY_HOST_TO_DEVICE))) {
            break;
        }
        if (!CHECK_ACL(aclrtMemset(recv_buf, recv_bytes, 0, recv_bytes))) {
            break;
        }
        log_rank(rank, "input copy OK");

        log_rank(rank, "HcclAllGather begin count=" + std::to_string(count));
        if (!CHECK_HCCL(HcclAllGather(send_buf, recv_buf, count, HCCL_DATA_TYPE_INT32, comm, stream))) {
            break;
        }
        log_rank(rank, "HcclAllGather launched");
        log_rank(rank, "aclrtSynchronizeStream begin");
        if (!CHECK_ACL(aclrtSynchronizeStream(stream))) {
            break;
        }
        log_rank(rank, "aclrtSynchronizeStream OK");
        log_rank(rank, "output copy begin");
        if (!CHECK_ACL(aclrtMemcpy(host_recv.data(), recv_bytes, recv_buf, recv_bytes, ACL_MEMCPY_DEVICE_TO_HOST))) {
            break;
        }
        log_rank(rank, "output copy OK");

        for (int src_rank = 0; src_rank < world_size; ++src_rank) {
            for (uint64_t i = 0; i < count; ++i) {
                const auto got = host_recv[static_cast<uint64_t>(src_rank) * count + i];
                if (got != src_rank) {
                    std::cerr << "[rank " << rank << "] verify failed at src_rank=" << src_rank << " offset=" << i
                              << " expected=" << src_rank << " got=" << got << "\n";
                    rc = 1;
                    goto cleanup;
                }
            }
        }

        log_rank(rank, "AllGather verify OK");
        rc = 0;
    } while (false);

cleanup:
    log_rank(rank, "cleanup begin rc=" + std::to_string(rc));
    if (stream != nullptr) {
        aclrtSynchronizeStream(stream);
    }
    if (recv_buf != nullptr) {
        aclrtFree(recv_buf);
    }
    if (send_buf != nullptr) {
        aclrtFree(send_buf);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (comm != nullptr) {
        HcclCommDestroy(comm);
    }
    if (device_set) {
        aclrtResetDevice(device_id);
    }
    if (acl_inited) {
        aclFinalize();
    }

    log_rank(rank, "cleanup OK; entering MPI_Barrier");
    MPI_Barrier(MPI_COMM_WORLD);
    log_rank(rank, "MPI_Barrier OK");
    if (rank == 0 && rc == 0) {
        std::cout << "A3 HCCL AllGather smoke PASS" << std::endl;
    }
    MPI_Finalize();
    return rc;
}
