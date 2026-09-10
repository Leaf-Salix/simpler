# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware UT for the Python kernel-mode surface of ChipWorker.

The Python twin of tests/ut/cpp/hardware/test_kernel_mode_entry.cpp, and the
inverse of test_platform_comm.py's contract: there ChipWorker owns ACL bring-up
and stream lifetime internally, here the *caller* owns both and lends the stream
in. That inversion is what kernel mode is, so the test does its own device bind
and stream creation through ``_acl_bind_device`` / ``_acl_create_stream`` and
hands the resulting integer address to kernel_launch. In production that integer
comes from the framework instead — torch_npu.npu.current_stream().npu_stream.

TMR init and prepare succeed on private streams. Execution remains unsupported
until the binding/executor providers exist, so direct launch must reach the C ABI
and return INVALID_STATE. The caller stream must survive successful finalization.

Each case runs in a forked subprocess: kernel_init binds a runtime library into
the process and the ACL device bind is per-thread, so a fresh process per case
keeps one case's state out of the next.

The ``runtime`` marker is what makes conftest's resource phase dispatch these
rather than deselect them, so it is load-bearing rather than descriptive — the
runtime it names is the one whose binaries the cases load.
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
import traceback
from pathlib import Path

import pytest


def _finalize_worker(worker, result) -> None:
    if worker is not None:
        try:
            worker.finalize()
        except Exception as exc:  # noqa: BLE001
            result["ok"] = False
            result["finalize_error"] = str(exc)


def _run_case(case: str, device_id: int, platform: str, queue) -> None:
    """Subprocess body: stand up the caller's device + stream, then drive the
    kernel-mode surface and report what came back."""
    result: dict[str, object] = {"case": case, "stage": "start", "ok": False}
    stream = 0
    worker = None
    try:
        import _task_interface as native
        from simpler.task_interface import CallConfig, ChipCallable, ChipStorageTaskArgs, ChipWorker

        from simpler_setup.kernel_compiler import KernelCompiler
        from simpler_setup.runtime_builder import RuntimeBuilder

        bins = RuntimeBuilder(platform=platform).get_binaries("tensormap_and_ringbuffer", build=False)

        # The caller's device and stream. simpler must not create, reset or
        # destroy any of this.
        if case != "program_init_still_works":
            native._acl_bind_device(device_id)
            stream = native._acl_create_stream()
            result["stream_nonzero"] = bool(stream)
            result["stage"] = "borrowed"

        worker = ChipWorker()
        config = CallConfig()

        if case == "init_prepare_launch_close":
            worker.kernel_init(device_id, bins, config)
            assert worker._impl.initialized
            assert not worker.kernel_mode_supported
            with tempfile.TemporaryDirectory(prefix="kernel-entry-") as build_dir:
                binary = KernelCompiler(platform).compile_orchestration(
                    "tensormap_and_ringbuffer",
                    str(Path(__file__).resolve().parents[1] / "kernel_prepare_orchestration.cpp"),
                    build_dir=build_dir,
                )
            chip = ChipCallable.build(
                signature=[], func_name="kernel_prepare_orchestration", binary=binary, children=[]
            )
            callable_id = worker.kernel_prepare_callable(chip)
            assert callable_id == 0
            assert worker._callable_registry[callable_id] is chip
            with pytest.raises(RuntimeError, match="-1003") as excinfo:
                worker.kernel_launch(callable_id, ChipStorageTaskArgs(), stream)
            result["error"] = str(excinfo.value)
            worker.finalize()
            assert not worker._impl.initialized
            assert not worker._callable_registry
            native._acl_destroy_stream(stream)
            stream = 0
            result["stream_destroyed_by_caller"] = True
            result["ok"] = True

        elif case == "null_stream_rejected":
            with pytest.raises(ValueError, match="caller_stream") as excinfo:
                worker.kernel_launch(0, ChipStorageTaskArgs(), 0)
            result["error"] = str(excinfo.value)
            result["ok"] = "caller_stream" in str(excinfo.value)

        elif case == "generation_is_unique":
            first = native._ChipWorker.next_kernel_context_generation()
            second = native._ChipWorker.next_kernel_context_generation()
            result["first"], result["second"] = first, second
            result["ok"] = first != 0 and second > first

        elif case == "uninitialized_surface_refuses":
            errors = []
            for label, fn in (
                ("supported", lambda: worker.kernel_mode_supported),
                ("launch", lambda: worker.kernel_launch(0, ChipStorageTaskArgs(), stream)),
            ):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    errors.append(label)
                    result[f"err_{label}"] = str(exc)
            result["refused"] = errors
            result["ok"] = errors == ["supported", "launch"]

        elif case == "program_init_still_works":
            # The program path must be unchanged by the kernel additions, and
            # the two identities must stay mutually exclusive on one worker.
            worker.init(device_id=device_id, bins=bins)
            result["stage"] = "program_init"
            result["initialized"] = bool(worker._impl.initialized)
            result["kernel_supported"] = bool(worker.kernel_mode_supported)
            try:
                worker.kernel_init(device_id, bins, config)
                result["second_init_refused"] = False
            except RuntimeError:
                result["second_init_refused"] = True
            worker.finalize()
            result["ok"] = result["initialized"] and not result["kernel_supported"] and result["second_init_refused"]

        else:
            raise AssertionError(f"unknown case {case}")

        result["stage"] = "done"
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        _finalize_worker(worker, result)
        if stream:
            try:
                import _task_interface as native

                native._acl_destroy_stream(stream)
                result["stream_destroyed_by_caller"] = True
            except Exception as exc:  # noqa: BLE001
                result["ok"] = False
                result["stream_destroy_error"] = str(exc)
        queue.put(result)


def _run_in_subprocess(case: str, device_id: int, platform: str) -> dict:
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_case, args=(case, device_id, platform, queue))
    proc.start()
    proc.join(timeout=300)
    assert proc.exitcode is not None, f"case {case} did not exit within 300s"
    assert not queue.empty(), f"case {case} produced no result (exitcode={proc.exitcode})"
    return queue.get()


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.parametrize(
    "case",
    [
        "init_prepare_launch_close",
        "null_stream_rejected",
        "generation_is_unique",
        "uninitialized_surface_refuses",
        "program_init_still_works",
    ],
)
def test_kernel_mode_surface_on_borrowed_stream(case, st_platform, st_device_ids):
    """Drive one kernel-mode case against a stream the test itself owns."""
    assert st_device_ids, "device_count(1) fixture must yield at least one id"
    result = _run_in_subprocess(case, int(st_device_ids[0]), st_platform)
    assert result["ok"], f"case {case} failed: {result}"


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_borrowed_stream_survives_kernel_finalize(st_platform, st_device_ids):
    """Successful finalization releases only the context-owned resources."""
    assert st_device_ids
    result = _run_in_subprocess("init_prepare_launch_close", int(st_device_ids[0]), st_platform)
    assert result.get("stream_nonzero"), f"test never obtained a stream: {result}"
    assert result["ok"], f"kernel lifecycle did not behave per contract: {result}"
    assert result.get("stream_destroyed_by_caller"), f"finalize invalidated the borrowed stream: {result}"
