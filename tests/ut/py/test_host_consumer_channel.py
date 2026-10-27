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
import socket
import threading
import time
from typing import Any

import pytest
from simpler import host_consumer as hc
from simpler.buffer import AccessMode


def _options(**kwargs) -> dict[str, Any]:
    return dict(nbytes=64, capacity=32, access=AccessMode.READWRITE, timeout_s=2.0, **kwargs)


def _unexpected_transfer(*_args):
    pytest.fail("unexpected backend transfer")


def _run(fn, transfer=_unexpected_transfer, **overrides):
    options = _options()
    options.update(overrides)
    hc.serve_host_call(
        lambda address, nonce: hc.run_host_call(address, nonce, fn, **options),
        transfer,
        expected_pid=os.getpid(),
        **options,
    )


def _process_call(pipe):
    pipe.send(("ready", os.getpid()))
    address, nonce = pipe.recv()

    def fn(payload):
        payload.write(8, bytes(range(16)))
        result = bytearray(16)
        payload.read(8, result)
        assert result == bytes(range(16))

    try:
        hc.run_host_call(address, nonce, fn, **_options())
        pipe.send(("ok", os.getpid()))
    except BaseException as error:  # noqa: BLE001
        pipe.send(("error", repr(error)))
    finally:
        pipe.close()


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_separate_process_round_trip_preserves_guards_and_serving_thread(method):
    ctx = multiprocessing.get_context(method)
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_process_call, args=(child,))
    process.start()
    child.close()
    storage, calls = bytearray([0xCC] * 64), []
    serving_thread = threading.get_ident()

    def transfer(writing, offset, data):
        assert threading.get_ident() == serving_thread
        calls.append(writing)
        if writing:
            storage[offset : offset + len(data)] = data
            return b""
        return bytes(storage[offset : offset + data])

    def control(address, nonce):
        parent.send((address, nonce))
        assert parent.poll(3)
        assert parent.recv() == ("ok", process.pid)
        process.join(3)
        assert process.exitcode == 0

    try:
        assert parent.poll(10)
        assert parent.recv() == ("ready", process.pid)
        hc.serve_host_call(control, transfer, expected_pid=process.pid, **_options())
        assert process.pid != os.getpid() and not process.is_alive()
        assert calls == [True, False]
        assert storage == bytes([0xCC] * 8) + bytes(range(16)) + bytes([0xCC] * 40)
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(3)


def test_zero_io_call_settles_and_escaped_payload_is_revoked():
    escaped = []

    def control(address, nonce):
        hc.run_host_call(address, nonce, escaped.append, **_options())
        with pytest.raises(RuntimeError, match="active call thread"):
            escaped[0].read(0, bytearray(1))

    hc.serve_host_call(control, _unexpected_transfer, expected_pid=os.getpid(), **_options())
    with pytest.raises(RuntimeError, match="active call thread"):
        escaped[0].read(0, bytearray(1))


@pytest.mark.parametrize("access,operation", [(AccessMode.READ, "write"), (AccessMode.WRITE, "read")])
def test_client_permissions_fail_before_transfer(access, operation):
    def fn(payload):
        with pytest.raises(PermissionError, match="not granted"):
            getattr(payload, operation)(0, bytearray(1))

    _run(fn, access=access)


def test_capacity_range_and_request_count_are_bounded():
    def fn(payload):
        with pytest.raises(ValueError, match="capacity"):
            payload.write(0, bytes(33))
        with pytest.raises(ValueError, match="exceeds part"):
            payload.read(63, bytearray(2))
        payload.write(0, b"x")
        with pytest.raises(ValueError, match="request limit"):
            payload.write(0, b"x")

    _run(fn, lambda *_: b"", request_limit=1)


@pytest.mark.parametrize("stage", ["before-connect", "partial-nonce", "partial-header"])
def test_control_error_wakes_every_receive_stage_and_preserves_original(stage):
    error = ValueError("native control failed")

    def control(address, nonce):
        if stage == "before-connect":
            raise error
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(1)
            sock.connect(address)
            if stage == "partial-nonce":
                sock.sendall(nonce[:1])
            else:
                channel = hc._Channel(sock, time.monotonic() + 1)
                channel.send(nonce)
                channel.recv(hc._HEADER.size)
                channel.send(b"\1")
            raise error

    with pytest.raises(ValueError) as exc:
        hc.serve_host_call(control, _unexpected_transfer, expected_pid=os.getpid(), **_options())
    assert exc.value is error


def test_callable_exception_is_not_hidden_by_done_or_channel_cleanup():
    error = LookupError("child callable failed")

    def fn(_payload):
        raise error

    with pytest.raises(LookupError) as exc:
        _run(fn)
    assert exc.value is error


def test_backend_error_reaches_child_and_does_not_return_success():
    def transfer(*_args):
        raise ValueError("backend copy failed")

    with pytest.raises(RuntimeError, match="backend copy failed"):
        _run(lambda payload: payload.write(0, b"x"), transfer)


@pytest.mark.parametrize("fault", ["nonce", "eof", "sequence", "capacity", "permission"])
def test_server_revalidates_untrusted_local_requests(fault):
    def control(address, nonce):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(1)
            sock.connect(address)
            channel = hc._Channel(sock, time.monotonic() + 1)
            channel.send(b"x" * 16 if fault == "nonce" else nonce)
            if fault == "nonce":
                with pytest.raises(EOFError):
                    channel.recv(1)
                return
            assert hc._HEADER.unpack(channel.recv(hc._HEADER.size)) == (hc._OK, 0, 0, 0)
            if fault == "eof":
                return
            seq, size = (2 if fault == "sequence" else 1), (33 if fault == "capacity" else 1)
            channel.send(hc._HEADER.pack(hc._WRITE, seq, 0, size))
            if fault == "sequence":
                with pytest.raises(EOFError):
                    channel.recv(1)
            else:
                op, received_seq, _, length = hc._HEADER.unpack(channel.recv(hc._HEADER.size))
                assert op == hc._ERROR and received_seq == seq and length <= hc._ERROR_BYTES
                assert channel.recv(length)

    options = _options()
    if fault == "permission":
        options["access"] = AccessMode.READ
    with pytest.raises((PermissionError, EOFError, ValueError, hc._ControlFinished)):
        hc.serve_host_call(control, _unexpected_transfer, expected_pid=os.getpid(), **options)


def test_duplicate_sequence_cannot_repeat_a_transfer():
    calls = []

    def transfer(writing, offset, size):
        calls.append((writing, offset, size))
        return b"x"

    def control(address, nonce):
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(1)
            sock.connect(address)
            channel = hc._Channel(sock, time.monotonic() + 1)
            channel.send(nonce)
            channel.recv(hc._HEADER.size)
            request = hc._HEADER.pack(hc._READ, 1, 0, 1)
            channel.send(request)
            assert hc._HEADER.unpack(channel.recv(hc._HEADER.size)) == (hc._OK, 1, 0, 1)
            assert channel.recv(1) == b"x"
            channel.send(request)
            with pytest.raises(EOFError):
                channel.recv(1)

    with pytest.raises(ValueError, match="sequence mismatch"):
        hc.serve_host_call(control, transfer, expected_pid=os.getpid(), **_options())
    assert calls == [(False, 0, 1)]


def test_unsettled_control_is_reported_at_absolute_deadline():
    release, exited = threading.Event(), threading.Event()

    def control(_address, _nonce):
        try:
            assert release.wait(3)
        finally:
            exited.set()

    options = _options()
    options["timeout_s"] = 0.1
    started = time.monotonic()
    try:
        with pytest.raises(hc.HostCallUnsettledError) as exc:
            hc.serve_host_call(control, _unexpected_transfer, expected_pid=os.getpid(), **options)
        assert time.monotonic() - started < 1 and not exited.is_set()
        assert exc.value.control_thread.is_alive()
    finally:
        release.set()
        assert exited.wait(1)
    exc.value.control_thread.join(1)
    assert not exc.value.control_thread.is_alive()


@pytest.mark.skipif(not hasattr(socket, "SO_PEERCRED"), reason="Linux peer credentials")
def test_linux_peer_pid_must_match_directed_child():
    with pytest.raises((PermissionError, EOFError, OSError)):
        hc.serve_host_call(
            lambda address, nonce: hc.run_host_call(address, nonce, lambda _: None, **_options()),
            _unexpected_transfer,
            expected_pid=os.getpid() + 1,
            **_options(),
        )
