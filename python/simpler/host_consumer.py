# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-call local Host payload transport; transfers run on the serving thread."""

from __future__ import annotations

import contextlib
import ctypes
import hmac
import math
import operator
import os
import secrets
import select
import socket
import struct
import tempfile
import threading
import time

from .buffer import AccessMode
from .comm_region import PayloadPart, RegionPartSpan

_HEADER = struct.Struct("!BIQI")
_OK, _READ, _WRITE, _DONE, _ERROR = range(5)
_ERROR_BYTES = 4096


class HostCallUnsettledError(RuntimeError):
    """The native control thread is still alive; its transport must be retained."""

    def __init__(self, thread):
        super().__init__("Host call control remains unsettled after its deadline")
        self.control_thread = thread


class _ControlFinished(RuntimeError):
    pass


def _deadline(nbytes, capacity, access, timeout_s, request_limit):
    nbytes, capacity, request_limit = map(operator.index, (nbytes, capacity, request_limit))
    if not 0 < nbytes < 2**64 or not 0 < capacity < 2**32 or not 0 < request_limit < 2**32 - 1:
        raise ValueError("invalid Host payload extent, capacity or request limit")
    if access not in (AccessMode.READ, AccessMode.WRITE, AccessMode.READWRITE):
        raise ValueError("invalid Host payload access")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Host call timeout must be finite and positive")
    return time.monotonic() + timeout_s


class _Channel:
    def __init__(self, sock, deadline, wake=None):
        self.sock, self.deadline, self.wake = sock, deadline, wake
        sock.setblocking(False)

    def wait(self, *, writing=False):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Host call deadline expired")
        readers = ([] if writing else [self.sock]) + ([] if self.wake is None else [self.wake])
        readable, writable, _ = select.select(readers, [self.sock] if writing else [], [], remaining)
        if self.sock in writable or self.sock in readable:
            return
        if self.wake in readable:
            raise _ControlFinished("Host control finished before the payload session completed")
        raise TimeoutError("Host call deadline expired")

    def send(self, data):
        data = memoryview(data)
        while data:
            self.wait(writing=True)
            try:
                size = self.sock.send(data)
            except BlockingIOError:
                continue
            if not size:
                raise EOFError("Host payload peer closed during send")
            data = data[size:]

    def recv(self, size):
        data = bytearray()
        while len(data) < size:
            self.wait()
            try:
                chunk = self.sock.recv(size - len(data))
            except BlockingIOError:
                continue
            if not chunk:
                raise EOFError("Host payload peer closed before DONE")
            data.extend(chunk)
        return bytes(data)

    def frame(self, op, seq, offset=0, data=b""):
        self.send(_HEADER.pack(op, seq, offset, len(data)))
        self.send(data)


def _validate_io(op, offset, size, nbytes, capacity, access):
    RegionPartSpan(0, nbytes).validate_range(offset, size, "Host payload")
    if size > capacity:
        raise ValueError("Host payload request exceeds staging capacity")
    if (op == _READ and access not in (AccessMode.READ, AccessMode.READWRITE)) or (
        op == _WRITE and access not in (AccessMode.WRITE, AccessMode.READWRITE)
    ):
        raise PermissionError("Host payload access is not granted")


def _serve(channel, transfer, *, nbytes, capacity, access, request_limit, finished):
    for seq in range(1, request_limit + 2):
        op, received_seq, offset, size = _HEADER.unpack(channel.recv(_HEADER.size))
        if received_seq != seq:
            raise ValueError("Host payload request sequence mismatch")
        if op == _DONE and offset == size == 0:
            return
        try:
            if op not in (_READ, _WRITE) or seq > request_limit:
                raise ValueError("Host payload request is not admitted")
            _validate_io(op, offset, size, nbytes, capacity, access)
            value = channel.recv(size) if op == _WRITE else size
            if finished.is_set():
                raise RuntimeError("Host control finished before its payload transfer")
            result = transfer(op == _WRITE, offset, value)
            if not isinstance(result, bytes) or len(result) != (size if op == _READ else 0):
                raise RuntimeError("Host payload transfer returned an invalid result")
            channel.frame(_OK, seq, data=result)
        except BaseException as error:
            with contextlib.suppress(Exception):
                channel.frame(_ERROR, seq, data=str(error).encode()[:_ERROR_BYTES])
            raise
    raise ValueError("Host payload request limit exceeded")


def serve_host_call(control, transfer, *, nbytes, capacity, access, timeout_s, expected_pid, request_limit=64):
    """Serve one child call and join its control thread before successful return.

    Linux authenticates the peer PID/UID against expected_pid. Other
    platforms use the private directory and per-call nonce as the capability.
    """
    deadline = _deadline(nbytes, capacity, access, timeout_s, request_limit)
    if operator.index(expected_pid) <= 0:
        raise ValueError("Host payload requires a positive child PID")
    errors, finished = [], threading.Event()
    with tempfile.TemporaryDirectory(prefix="shc-") as directory, socket.socket(socket.AF_UNIX) as listener:
        os.chmod(directory, 0o700)
        address, nonce = os.path.join(directory, "call"), secrets.token_bytes(16)
        listener.bind(address)
        listener.listen(1)
        wake_rx, wake_tx = socket.socketpair()
        with wake_rx, wake_tx:

            def invoke():
                try:
                    control(address, nonce)
                except BaseException as error:
                    errors.append(error)
                finally:
                    finished.set()
                    with contextlib.suppress(OSError):
                        wake_tx.send(b"\0")

            thread = threading.Thread(target=invoke, name="host-consumer-control", daemon=True)
            thread.start()
            failure = None
            try:
                _Channel(listener, deadline, wake_rx).wait()
                peer, _ = listener.accept()
                with peer:
                    if hasattr(socket, "SO_PEERCRED"):
                        pid, uid, _ = struct.unpack("3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if pid != expected_pid or uid != os.getuid():
                            raise PermissionError("Host payload peer identity mismatch")
                    channel = _Channel(peer, deadline, wake_rx)
                    if not hmac.compare_digest(channel.recv(16), nonce):
                        raise PermissionError("Host payload authentication failed")
                    listener.close()
                    channel.frame(_OK, 0)
                    _serve(
                        channel,
                        transfer,
                        nbytes=nbytes,
                        capacity=capacity,
                        access=access,
                        request_limit=request_limit,
                        finished=finished,
                    )
            except BaseException as error:
                failure = error
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive():
                raise HostCallUnsettledError(thread) from failure
            if errors:
                raise errors[0]
            if failure is not None:
                raise failure


class _PayloadAccess:
    def __init__(self, channel, *, nbytes, capacity, access, request_limit):
        self.channel, self.nbytes, self.capacity, self.access = channel, nbytes, capacity, access
        self.request_limit, self.seq, self.live = request_limit, 0, True
        self.thread_id = threading.get_ident()

    def exchange(self, op, offset, pointer, size):
        if not self.live or threading.get_ident() != self.thread_id:
            raise RuntimeError("Host payload is only valid on its active call thread")
        _validate_io(op, offset, size, self.nbytes, self.capacity, self.access)
        if self.seq >= self.request_limit:
            raise ValueError("Host payload request limit exceeded")
        self.seq += 1
        self.channel.send(_HEADER.pack(op, self.seq, offset, size))
        if op == _WRITE:
            self.channel.send(ctypes.string_at(pointer, size))
        response, seq, offset, length = _HEADER.unpack(self.channel.recv(_HEADER.size))
        if seq != self.seq or offset or response not in (_OK, _ERROR):
            raise ValueError("invalid Host payload reply")
        if length > (_ERROR_BYTES if response == _ERROR else size if op == _READ else 0):
            raise ValueError("Host payload reply exceeds expected size")
        result = self.channel.recv(length)
        if response == _ERROR:
            raise RuntimeError(result.decode(errors="replace"))
        if length != (size if op == _READ else 0):
            raise ValueError("Host payload reply size mismatch")
        if op == _READ:
            ctypes.memmove(pointer, result, size)

    def write_bytes(self, _span, offset, pointer, size):
        self.exchange(_WRITE, offset, pointer, size)

    def read_bytes(self, _span, offset, pointer, size):
        self.exchange(_READ, offset, pointer, size)


def run_host_call(address, nonce, fn, *, nbytes, capacity, access, timeout_s, request_limit=64):
    """Run a child callable with a revocable, synchronous PayloadPart."""
    deadline = _deadline(nbytes, capacity, access, timeout_s, request_limit)
    if not isinstance(nonce, bytes) or len(nonce) != 16:
        raise ValueError("invalid Host payload nonce")
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(max(0, deadline - time.monotonic()))
        sock.connect(address)
        channel = _Channel(sock, deadline)
        channel.send(nonce)
        if _HEADER.unpack(channel.recv(_HEADER.size)) != (_OK, 0, 0, 0):
            raise ValueError("invalid Host payload handshake")
        backend = _PayloadAccess(channel, nbytes=nbytes, capacity=capacity, access=access, request_limit=request_limit)
        completed = False
        try:
            fn(PayloadPart(RegionPartSpan(0, nbytes), backend))
            completed = True
        finally:
            backend.live = False
            if completed:
                channel.frame(_DONE, backend.seq + 1)
            else:
                with contextlib.suppress(Exception):
                    channel.frame(_DONE, backend.seq + 1)
