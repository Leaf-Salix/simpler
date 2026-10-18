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

#include <gtest/gtest.h>

#include "task_args_wire.h"

TEST(CommDomain, RefAndContextOnlyUseSurviveTheMailbox) {
    CommDomainRef ref{};
    ref.owner_instance_id[0] = 1;
    ref.domain_id = 42;
    ref.generation = 3;
    TaskArgs args;
    args.add_domain_ref(ref);
    args.add_domain_ref(ref);
    args.add_scalar(0x12345678);
    std::vector<uint8_t> bytes(task_args_blob_size(args));
    write_blob(bytes.data(), args);
    const auto view = read_blob(bytes.data(), bytes.size());
    ASSERT_EQ(view.domain_ref_count, 1);
    EXPECT_EQ(view.domain_ref(0), ref);
    EXPECT_EQ(view.scalars[0], 0x12345678);
    args.clear();
    EXPECT_TRUE(args.domain_refs().empty());
}

TEST(CommDomain, DomainDescriptorCannotContainAnUninitializedRef) {
    BufferDescriptor desc{};
    desc.magic = BUFFER_DESCRIPTOR_MAGIC;
    desc.address_space = static_cast<uint8_t>(AddressSpace::DEVICE);
    desc.access = static_cast<uint8_t>(AccessMode::READWRITE);
    desc.backend_kind = static_cast<uint8_t>(BackendKind::DOMAIN_WINDOW);
    desc.identity.generation = 1;
    desc.nbytes = 64;
    desc.body_len = DESC_MAX_BYTES;
    EXPECT_THROW(validate_buffer_descriptor(desc), std::invalid_argument);
}

TEST(CommDomain, BlobRejectsMissingReferenceBytesAndReservedFields) {
    TaskArgs args;
    CommDomainRef ref{};
    ref.domain_id = 1;
    ref.generation = 1;
    args.add_domain_ref(ref);
    std::vector<uint8_t> bytes(task_args_blob_size(args));
    write_blob(bytes.data(), args);
    EXPECT_THROW(read_blob(bytes.data(), bytes.size() - 1), std::runtime_error);
    bytes[12] = 1;
    EXPECT_THROW(read_blob(bytes.data(), bytes.size()), std::runtime_error);
}

TEST(CommDomain, DomainTensorRequiresItsDeclaredReferenceOnReceive) {
    Tensor tensor{};
    auto &desc = tensor.buffer;
    desc.magic = BUFFER_DESCRIPTOR_MAGIC;
    desc.address_space = static_cast<uint8_t>(AddressSpace::DEVICE);
    desc.access = static_cast<uint8_t>(AccessMode::READWRITE);
    desc.backend_kind = static_cast<uint8_t>(BackendKind::DOMAIN_WINDOW);
    desc.identity.owner_instance_id[0] = 1;
    desc.identity.buffer_id = 1;
    desc.identity.generation = 1;
    desc.nbytes = 64;
    desc.body_len = 32;
    desc.body[0] = 1;
    domain_write_le(desc.body + 8, 42, 8);
    domain_write_le(desc.body + 16, 1, 4);
    tensor.ndims = 1;
    tensor.shapes[0] = 16;
    tensor.strides[0] = 1;
    tensor.dtype = DataType::FLOAT32;
    TaskArgs args;
    args.add_tensor(tensor);
    std::vector<uint8_t> bytes(task_args_blob_size(args));
    write_blob(bytes.data(), args);
    EXPECT_NO_THROW(read_blob(bytes.data(), bytes.size()).tensors(0));
    bytes[8] = 0;
    EXPECT_THROW(read_blob(bytes.data(), bytes.size()).tensors(0), std::invalid_argument);
    bytes[8] = 1;
    bytes[TASK_ARGS_BLOB_HEADER_SIZE + sizeof(Tensor) + 8] = 43;
    EXPECT_THROW(read_blob(bytes.data(), bytes.size()).tensors(0), std::invalid_argument);
}
