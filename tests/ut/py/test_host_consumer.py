# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import multiprocessing
import os
import threading

import pytest
from simpler.buffer import AccessMode
from simpler.host_consumer import run_host_call, serve_host_call


def _client(connection, mode):
    address, nonce = connection.recv()
    connection.close()
    saved = []

    def consume(payload):
        assert os.getpid() != os.getppid()
        saved.append(payload)
        if mode == "readonly":
            with pytest.raises(PermissionError):
                payload.write(10, b"SUB!")
        elif mode == "bounds":
            with pytest.raises(ValueError):
                payload.read(0, bytearray(9))
            with pytest.raises(ValueError):
                payload.read(31, bytearray(2))
            with pytest.raises(ValueError):
                payload.write(31, b"xx")
        observed = bytearray(8)
        payload.read(8, observed)
        assert observed == bytes(range(8, 16))
        if mode == "write":
            payload.write(10, b"SUB!")
            payload.read(8, observed)
            assert observed == bytes((8, 9)) + b"SUB!" + bytes((14, 15))

    run_host_call(
        address,
        nonce,
        consume,
        nbytes=32,
        capacity=8,
        access=AccessMode.READ if mode == "readonly" else AccessMode.READWRITE,
        timeout_s=10,
        request_limit=3,
    )
    with pytest.raises(RuntimeError, match="active call"):
        saved[0].read(0, bytearray(1))


@pytest.mark.parametrize("mode", ["write", "readonly", "bounds"])
def test_real_process_payload_transfer_preserves_owner_thread_and_bounds(mode):
    data = bytearray(range(32))
    transfers = []
    owner_thread = threading.get_ident()
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_client, args=(receive, mode))
    process.start()
    receive.close()

    def control(address, nonce):
        send.send((address, nonce))
        send.close()
        process.join(10)
        assert process.exitcode == 0, f"Host child failed or did not exit: {process.exitcode}"

    def transfer(writing, offset, value):
        transfers.append((writing, threading.get_ident()))
        if writing:
            data[offset : offset + len(value)] = value
            return b""
        return bytes(data[offset : offset + value])

    try:
        serve_host_call(
            control,
            transfer,
            nbytes=32,
            capacity=8,
            access=AccessMode.READ if mode == "readonly" else AccessMode.READWRITE,
            timeout_s=10,
            request_limit=3,
            expected_pid=process.pid,
        )
    finally:
        send.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        process.close()
    expected = bytearray(range(32))
    if mode == "write":
        expected[10:14] = b"SUB!"
    assert data == expected
    assert [writing for writing, _ in transfers] == ([False, True, False] if mode == "write" else [False])
    assert all(thread == owner_thread for _, thread in transfers)
