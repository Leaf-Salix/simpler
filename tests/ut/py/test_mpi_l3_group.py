# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MPI L3 group activation and transport-isolation tests."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import simpler.mpi_group_mailbox as mailbox_mod
import simpler.worker as worker_mod
from simpler import mpi_l3_session
from simpler.mpi_group_mailbox import MAILBOX_SIZE, MailboxGroupState, MpiRankError
from simpler.remote_l3_protocol import ControlName
from simpler.worker import MpiL3GroupSpec, Worker


class _ReadyMailbox:
    def __init__(self, world_size: int) -> None:
        self.world_size = int(world_size)
        self.group_state = MailboxGroupState.READY
        self.address = 0x12340000
        self.closed = False
        self.terminal_messages: list[str] = []

    def manifest(self):
        return {
            "name": "pytest-mpi-mailbox",
            "mailbox_bytes": MAILBOX_SIZE,
            "world_size": self.world_size,
        }

    def terminal_reason(self):
        return self.terminal_messages[-1] if self.terminal_messages else ""

    def mark_terminal(self, message):
        self.group_state = MailboxGroupState.TERMINAL
        self.terminal_messages.append(str(message))

    def close(self, *, unlink=False):
        assert unlink
        self.closed = True


class _FakeProcess:
    pid = 4242
    returncode = None

    def poll(self):
        return self.returncode


class _InertThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def start(self):
        return None

    def join(self, timeout=None):
        return None


def _mpi_worker() -> Worker:
    worker = Worker(level=4, num_sub_workers=0)
    worker.add_mpirun_worker_group(
        MpiL3GroupSpec(
            hosts=("127.0.0.1", "127.0.0.1"),
            platform="a2a3sim",
            device_ids_by_rank=((0,), (1,)),
            global_device_ranks_by_rank=((0,), (1,)),
        )
    )
    return worker


def test_mpi_activation_attaches_only_mailbox_endpoint(monkeypatch):
    worker = _mpi_worker()
    mailbox = _ReadyMailbox(world_size=2)
    native_worker = MagicMock()
    worker._worker = native_worker
    monkeypatch.setattr(mailbox_mod.MpiGroupMailbox, "create", lambda **_kwargs: mailbox)
    monkeypatch.setattr(worker_mod.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProcess())
    monkeypatch.setattr(worker_mod.threading, "Thread", _InertThread)
    try:
        worker._activate_mpirun_worker_groups(worker_mod.time.monotonic() + 10.0)
        native_worker.add_mpi_group_mailbox.assert_called_once()
        native_worker.add_remote_l3_socket.assert_not_called()
        group = worker._mpi_l3_groups[0]
        assert group.manifest_path is not None
        with open(group.manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        assert manifest["mailbox"]["name"] == "pytest-mpi-mailbox"
        assert "command_port" not in json.dumps(manifest)
        assert "health_port" not in json.dumps(manifest)
        assert "listen_host" not in json.dumps(manifest)
        assert "connect_host" not in json.dumps(manifest)
    finally:
        group = worker._mpi_l3_groups[0]
        group.process = None
        group.monitor_thread = None
        worker._worker = None
        worker.close()
        assert mailbox.closed


def test_unexpected_mpirun_exit_marks_group_terminal():
    worker = _mpi_worker()
    mailbox = _ReadyMailbox(world_size=2)
    group = worker._mpi_l3_groups[0]
    group.mailbox = mailbox
    process = _FakeProcess()
    process.returncode = 17
    process.wait = lambda: 17
    worker._monitor_mpirun_group(group, cast(Any, process))
    try:
        assert mailbox.group_state is MailboxGroupState.TERMINAL
        assert "status 17" in mailbox.terminal_reason()
    finally:
        group.process = None
        group.mailbox = None
        worker.close()


def test_mpi_spec_tcp_fields_are_optional_and_control_18_stays_reserved():
    spec = MpiL3GroupSpec(
        hosts=("127.0.0.1", "127.0.0.1"),
        platform="a2a3sim",
        device_ids_by_rank=((0,), (1,)),
    )
    assert spec.command_port_base is None
    assert spec.health_port_base is None
    assert spec.session_listen_hosts == ()
    assert spec.connect_hosts == ()
    assert worker_mod._CTRL_COMMITTED_DEVICE_MEMORY == 18
    assert 18 not in {int(control) for control in ControlName}


def test_mpi_rank_worker_uses_parent_topology_id_before_init(monkeypatch):
    captured = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self._topology_worker_id = None
            captured["worker"] = self

        def init(self, *args, **kwargs):
            raise RuntimeError("stop after worker construction")

        def close(self):
            pass

    class FailedStartupComm:
        def Get_rank(self):
            return 1

        def allgather(self, value):
            return [value]

        def bcast(self, value, root=0):
            return False, "stop after startup"

    monkeypatch.setattr(mpi_l3_session, "Worker", FakeWorker)
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_dispatcher_registry", lambda manifest: {})
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_inner_registry", lambda manifest, worker: {})

    rc = mpi_l3_session._run_group_session(
        dispatch_comm=FailedStartupComm(),
        domain_comm=None,
        group_manifest={"worker_ids": [3, 7]},
        manifest={
            "worker_id": 7,
            "platform": "a2a3sim",
            "runtime": "tensormap_and_ringbuffer",
            "device_ids": [1],
            "num_sub_workers": 0,
            "heap_ring_size": None,
            "startup_remaining_s": 1.0,
            "session_timeout_s": 1.0,
        },
    )

    assert rc == 1
    assert captured["worker"]._topology_worker_id == 7


def test_group_task_completion_preserves_errors_and_all_rank_settlement():
    import struct  # noqa: PLC0415

    from simpler.mpi_group_mailbox import MailboxOpcode, MailboxRequest, MailboxTarget  # noqa: PLC0415
    from simpler.remote_l3_protocol import (  # noqa: PLC0415
        FrameHeader,
        FrameType,
        decode_frame,
        encode_completion,
        encode_frame,
    )

    def frame(rank, *, error=0, settled=True):
        header = FrameHeader(frame_type=FrameType.COMPLETION, session_id=1, worker_id=rank + 10, sequence=8)
        return encode_frame(
            header, encode_completion(8, error, "kernel failed" if error else "", domain_use_settled=settled)
        )

    request_frame = encode_frame(FrameHeader(frame_type=FrameType.TASK, session_id=1, worker_id=10, sequence=5), b"")
    request = MailboxRequest(
        sequence_id=5, opcode=MailboxOpcode.TASK, target=MailboxTarget.GROUP, target_rank=-1, payloads=(request_frame,)
    )
    for rank_one_settled in (True, False):
        replies = mpi_l3_session._select_mailbox_replies(
            request, [(0, True, frame(0), None), (1, True, frame(1, error=7, settled=rank_one_settled), None)], (10, 11)
        )
        result = decode_frame(replies[0])
        sequence, error, size = struct.unpack_from("<QiI", result.payload)
        assert sequence == 5 and error == 7
        assert result.payload[16 : 16 + size] == b"MPI rank 1: kernel failed"
        assert result.payload[-1] == int(rank_one_settled)


@pytest.mark.parametrize("message", ["x" * 4096, "\u754c" * 1365 + "x"], ids=["ascii", "utf8"])
@pytest.mark.parametrize("settled", [False, True])
def test_group_completion_bounds_rank_prefix_by_utf8_bytes(message, settled):
    import struct  # noqa: PLC0415

    from simpler.mpi_group_mailbox import MailboxOpcode, MailboxRequest, MailboxTarget  # noqa: PLC0415
    from simpler.remote_l3_protocol import (  # noqa: PLC0415
        MAX_ERROR_BYTES,
        FrameHeader,
        FrameType,
        decode_frame,
        encode_completion,
        encode_frame,
    )

    request = MailboxRequest(
        sequence_id=5,
        opcode=MailboxOpcode.TASK,
        target=MailboxTarget.GROUP,
        target_rank=-1,
        payloads=(encode_frame(FrameHeader(frame_type=FrameType.TASK, session_id=1, worker_id=10, sequence=5), b""),),
    )
    gathered = []
    for rank in range(2):
        header = FrameHeader(frame_type=FrameType.COMPLETION, session_id=1, worker_id=rank + 10, sequence=8)
        payload = encode_completion(8, 7 if rank else 0, message if rank else "", domain_use_settled=settled)
        gathered.append((rank, True, encode_frame(header, payload), None))
    (reply,) = mpi_l3_session._select_mailbox_replies(request, gathered, (10, 11))
    result = decode_frame(reply)
    sequence, error, size = struct.unpack_from("<QiI", result.payload)
    text = result.payload[16 : 16 + size].decode("utf-8")
    assert sequence == 5 and error == 7
    assert text.startswith("MPI rank 1: ")
    assert len(text.encode("utf-8")) <= MAX_ERROR_BYTES
    assert message.startswith(text.removeprefix("MPI rank 1: "))
    assert result.payload[-1] == int(settled)


def test_group_task_completion_with_missing_or_malformed_rank_never_proves_settlement():
    import struct  # noqa: PLC0415

    from simpler.mpi_group_mailbox import MailboxOpcode, MailboxRequest, MailboxTarget  # noqa: PLC0415
    from simpler.remote_l3_protocol import (  # noqa: PLC0415
        FrameHeader,
        FrameType,
        decode_frame,
        encode_completion,
        encode_frame,
    )

    request_frame = encode_frame(FrameHeader(frame_type=FrameType.TASK, session_id=1, worker_id=10, sequence=5), b"")
    request = MailboxRequest(
        sequence_id=5, opcode=MailboxOpcode.TASK, target=MailboxTarget.GROUP, target_rank=-1, payloads=(request_frame,)
    )
    first = encode_frame(
        FrameHeader(frame_type=FrameType.COMPLETION, session_id=1, worker_id=10, sequence=8),
        encode_completion(8, 0, ""),
    )
    for worker_id, payload in ((11, b"\0" * 16), (10, encode_completion(8, 0, ""))):
        malformed = encode_frame(
            FrameHeader(frame_type=FrameType.COMPLETION, session_id=1, worker_id=worker_id, sequence=8), payload
        )
        gathered: list[tuple[int, bool, bytes, MpiRankError | None]] = [
            (0, True, first, None),
            (1, True, malformed, None),
        ]
        result = decode_frame(mpi_l3_session._select_mailbox_replies(request, gathered, (10, 11))[0])
        assert result.payload[-1] == 0
        assert struct.unpack_from("<i", result.payload, 8)[0] != 0
    result = decode_frame(mpi_l3_session._select_mailbox_replies(request, [(0, True, first, None)], (10, 11))[0])
    assert result.payload[-1] == 0
    assert b"missing a rank" in result.payload


@pytest.mark.parametrize("failure", ["aggregate", "publish", "report"])
def test_rank_zero_failure_still_broadcasts_group_termination(monkeypatch, failure):
    import queue  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    from simpler.mpi_group_mailbox import (  # noqa: PLC0415
        MailboxOpcode,
        MailboxRequest,
        MailboxRequestState,
        MailboxTarget,
    )
    from simpler.remote_l3_protocol import FrameHeader, FrameType, encode_completion, encode_frame  # noqa: PLC0415

    header = FrameHeader(frame_type=FrameType.TASK, session_id=1, worker_id=10, sequence=1)
    request = MailboxRequest(
        sequence_id=1,
        opcode=MailboxOpcode.TASK,
        target=MailboxTarget.GROUP,
        target_rank=-1,
        payloads=(encode_frame(header, b""),),
    )
    reply = encode_frame(
        FrameHeader(frame_type=FrameType.COMPLETION, session_id=1, worker_id=10, sequence=1),
        encode_completion(1, 0, ""),
    )
    broadcasts, errors, closed = [], [], []

    class Comm:
        def Get_rank(self):
            return 0

        def allgather(self, value):
            return [value]

        def gather(self, value, root=0):
            return [value]

        def bcast(self, value, root=0):
            broadcasts.append(value)
            return value

    class Mailbox:
        request_state = MailboxRequestState.REQUEST_READY
        group_state = MailboxGroupState.READY

        def publish_ready(self):
            pass

        def accept_request(self, **_kwargs):
            return request

        def complete_request(self, **_kwargs):
            raise RuntimeError("publish failed")

        def fail_request(self, **kwargs):
            errors.append(kwargs)
            if failure == "report":
                raise RuntimeError("report failed")

        def close(self):
            closed.append("mailbox")

    class Connection:
        def __init__(self):
            self.events = queue.Queue()

        def next_reply(self):
            return encode_frame(FrameHeader(frame_type=FrameType.HELLO, session_id=1, worker_id=10, sequence=0), b"")

        def exchange(self, _payload):
            return reply

    class Thread(_InertThread):
        def is_alive(self):
            return False

    worker = SimpleNamespace(init=lambda **_kw: None, close=lambda: closed.append("worker"))
    monkeypatch.setattr(mpi_l3_session, "Worker", lambda **_kw: worker)
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_dispatcher_registry", lambda _manifest: {})
    monkeypatch.setattr(mpi_l3_session, "_install_manifest_inner_registry", lambda *_args: {})
    monkeypatch.setattr(mpi_l3_session, "_InMemoryCommandConnection", Connection)
    monkeypatch.setattr(mpi_l3_session, "open_rank_mailbox", lambda *_args, **_kwargs: Mailbox())
    monkeypatch.setattr(mpi_l3_session.threading, "Thread", Thread)
    if failure != "publish":

        def fail_aggregate(*_args):
            raise RuntimeError("aggregate failed")

        monkeypatch.setattr(mpi_l3_session, "_select_mailbox_replies", fail_aggregate)

    def run():
        return mpi_l3_session._run_group_session(
            dispatch_comm=Comm(),
            domain_comm=Comm(),
            group_manifest={"worker_ids": [10], "mailbox": {}},
            manifest={"worker_id": 10, "session_id": 1, "platform": "a2a3sim", "session_timeout_s": 1},
        )

    if failure == "report":
        with pytest.raises(RuntimeError, match="report failed"):
            run()
    else:
        assert run() == 1
    assert broadcasts[-1] is True
    assert errors[0]["terminal"] and errors[0]["sequence_id"] == 1
    assert closed == ["worker", "mailbox"]
