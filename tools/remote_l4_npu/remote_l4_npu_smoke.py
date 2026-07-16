from __future__ import annotations

import argparse
import ctypes
import hashlib
from pathlib import Path

from simpler.callable_identity import build_chip_callable_descriptor, compute_callable_hashid, hashid_to_digest
from simpler.remote_l3_protocol import (
    CallableKind,
    ChipCallableBlobLocation,
    RemoteChipCallablePayload,
    encode_remote_chip_callable_payload,
)
from simpler.task_interface import CallConfig, ChipCallable, TaskArgs
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker
from simpler_setup.kernel_compiler import KernelCompiler

REMOTE_ORCH_TARGET = "tools.remote_l4_npu.remote_l4_npu_smoke:remote_noop_chip_orch"


def remote_noop_chip_orch(orch, args, cfg):
    """Remote L3 dispatcher callable imported by the session runner."""
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little", signed=False) for i in range(4))
    chip_handle = get_inner_handle(digest.hex())
    orch.submit_next_level(chip_handle, TaskArgs(), cfg, worker=0)


def _build_noop_chip_callable(platform: str, runtime: str) -> ChipCallable:
    kernel = Path(__file__).resolve().parent / "kernels" / "noop_orch.cpp"
    orch_binary = KernelCompiler(platform=platform).compile_orchestration(runtime, str(kernel))
    return ChipCallable.build(
        signature=[],
        func_name="aicpu_orchestration_entry",
        binary=orch_binary,
        children=[],
        config_name="aicpu_orchestration_config",
    )


def _remote_chip_register_payload(chip: ChipCallable, *, platform: str, runtime: str) -> tuple[bytes, bytes]:
    blob = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    descriptor = build_chip_callable_descriptor(target=chip, platform=platform, runtime=runtime)
    digest = hashid_to_digest(compute_callable_hashid(descriptor))
    payload = encode_remote_chip_callable_payload(
        RemoteChipCallablePayload(
            descriptor_bytes=descriptor,
            blob_location=ChipCallableBlobLocation.INLINE_BLOB,
            blob_size=len(blob),
            blob_sha256=hashlib.sha256(blob).digest(),
            inline_blob=blob,
            staged_blob_token=b"",
        )
    )
    return digest, payload


def _digest_args(digest: bytes) -> TaskArgs:
    if len(digest) != 32:
        raise ValueError("inner chip callable digest must be 32 bytes")
    args = TaskArgs()
    for offset in range(0, 32, 8):
        args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little", signed=False))
    return args


def _install_inner_chip_callable(worker: Worker, worker_id: int, digest: bytes, payload: bytes) -> None:
    assert worker._worker is not None  # noqa: SLF001
    result = worker._worker.remote_prepare_register(  # noqa: SLF001
        worker_id,
        "INNER_L3_WORKER",
        CallableKind.CHIP_CALLABLE.name,
        payload,
        digest,
    )
    if not result.ok:
        raise RuntimeError(f"remote prepare inner CHIP_CALLABLE failed on worker {worker_id}: {result.error_message}")
    result = worker._worker.remote_commit_register(  # noqa: SLF001
        worker_id,
        "INNER_L3_WORKER",
        CallableKind.CHIP_CALLABLE.name,
        digest,
    )
    if not result.ok:
        raise RuntimeError(f"remote commit inner CHIP_CALLABLE failed on worker {worker_id}: {result.error_message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an L4 -> two remote L3 -> NPU no-op smoke.")
    parser.add_argument("--machine-a", required=True, help="machineA daemon endpoint, host:port")
    parser.add_argument("--machine-b", required=True, help="machineB daemon endpoint, host:port")
    parser.add_argument("--machine-a-device", type=int, default=0)
    parser.add_argument("--machine-b-device", type=int, default=0)
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--session-timeout", type=float, default=60.0)
    parser.add_argument("--session-listen-host", default="0.0.0.0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    chip = _build_noop_chip_callable(args.platform, args.runtime)
    digest, payload = _remote_chip_register_payload(chip, platform=args.platform, runtime=args.runtime)

    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=args.session_timeout)
    try:
        worker_a = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=args.machine_a,
                platform=args.platform,
                runtime=args.runtime,
                device_ids=(args.machine_a_device,),
                transport="sim",
                session_listen_host=args.session_listen_host,
                allow_wildcard_session_bind=True,
            )
        )
        worker_b = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=args.machine_b,
                platform=args.platform,
                runtime=args.runtime,
                device_ids=(args.machine_b_device,),
                transport="sim",
                session_listen_host=args.session_listen_host,
                allow_wildcard_session_bind=True,
            )
        )
        remote_handle = worker.register(RemoteCallable(REMOTE_ORCH_TARGET), workers=[worker_a, worker_b])
        worker.init()
        worker._start_hierarchical()  # noqa: SLF001
        _install_inner_chip_callable(worker, worker_a, digest, payload)
        _install_inner_chip_callable(worker, worker_b, digest, payload)

        def parent_orch(orch, _args, cfg):
            orch.submit_next_level(remote_handle, _digest_args(digest), cfg, worker=worker_a)
            orch.submit_next_level(remote_handle, _digest_args(digest), cfg, worker=worker_b)

        worker.run(parent_orch, config=CallConfig())
        print(
            "remote L4 NPU smoke passed: "
            f"{args.machine_a}[device={args.machine_a_device}], "
            f"{args.machine_b}[device={args.machine_b_device}]"
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
