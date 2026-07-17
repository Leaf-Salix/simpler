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
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, RemoteTensorRef, TaskArgs
from simpler.task_interface import TensorArgType
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

REMOTE_ORCH_TARGET = "tools.remote_l4_npu.remote_l4_npu_smoke:remote_l3_group_orch"
ELEMENTS = 128 * 128
FLOAT_NBYTES = ctypes.sizeof(ctypes.c_float)
TENSOR_COUNT = 6
FloatArray = ctypes.c_float * ELEMENTS
_REMOTE_GROUP_KEEPALIVE = []


def remote_l3_group_orch(orch, args, cfg):
    """Remote L3 dispatcher: run test_l3_group's two-chip vector group."""
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    digest = b"".join(int(args.scalar(i)).to_bytes(8, "little", signed=False) for i in range(4))
    chip_handle = get_inner_handle(digest.hex())

    chip_args0 = TaskArgs()
    chip_args0.add_tensor(args.tensor(0), TensorArgType.INPUT)
    chip_args0.add_tensor(args.tensor(1), TensorArgType.INPUT)
    chip_args0.add_tensor(args.tensor(2), TensorArgType.OUTPUT_EXISTING)

    chip_args1 = TaskArgs()
    chip_args1.add_tensor(args.tensor(3), TensorArgType.INPUT)
    chip_args1.add_tensor(args.tensor(4), TensorArgType.INPUT)
    chip_args1.add_tensor(args.tensor(5), TensorArgType.OUTPUT_EXISTING)

    _REMOTE_GROUP_KEEPALIVE[:] = [chip_args0, chip_args1]
    orch.submit_next_level_group(chip_handle, [chip_args0, chip_args1], cfg)


def _build_vector_chip_callable(platform: str, runtime: str) -> ChipCallable:
    root = Path(__file__).resolve().parents[2]
    kernels = root / "examples" / "a2a3" / "tensormap_and_ringbuffer" / "vector_example" / "kernels"
    orch_src = kernels / "orchestration" / "example_orchestration.cpp"
    aiv_add = kernels / "aiv" / "kernel_add.cpp"
    aiv_add_scalar = kernels / "aiv" / "kernel_add_scalar.cpp"
    aiv_mul = kernels / "aiv" / "kernel_mul.cpp"

    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(runtime)
    orch_binary = kc.compile_orchestration(runtime, str(orch_src))

    def _aiv(path: Path) -> bytes:
        raw = kc.compile_incore(str(path), core_type="aiv", pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs)
        return raw if platform.endswith("sim") else extract_text_section(raw)

    children = [
        (0, CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv(aiv_add))),
        (1, CoreCallable.build(signature=[D.IN, D.OUT], binary=_aiv(aiv_add_scalar))),
        (2, CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv(aiv_mul))),
    ]
    return ChipCallable.build(
        signature=[D.IN, D.IN, D.OUT],
        func_name="aicpu_orchestration_entry",
        binary=orch_binary,
        children=children,
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


def _add_digest_scalars(task_args: TaskArgs, digest: bytes) -> None:
    if len(digest) != 32:
        raise ValueError("inner chip callable digest must be 32 bytes")
    for offset in range(0, 32, 8):
        task_args.add_scalar(int.from_bytes(digest[offset : offset + 8], "little", signed=False))


def _parse_device_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not ids:
        raise ValueError("device list must not be empty")
    return ids


def _make_array(value: float) -> FloatArray:
    arr = FloatArray()
    for idx in range(ELEMENTS):
        arr[idx] = value
    return arr


def _expected(lhs: float, rhs: float) -> float:
    summed = lhs + rhs
    return (summed + 1.0) * (summed + 2.0) + summed


def _make_remote_group_args(handles: list, digest: bytes) -> TaskArgs:
    args = TaskArgs()
    for index, handle in enumerate(handles):
        tag = TensorArgType.OUTPUT_EXISTING if index in (2, 5) else TensorArgType.INPUT
        args.add_tensor(RemoteTensorRef(handle, shape=(ELEMENTS,), dtype=DataType.FLOAT32), tag)
    _add_digest_scalars(args, digest)
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
    parser = argparse.ArgumentParser(description="Run L4 -> two remote L3 workers, each with a 2-NPU L3 group.")
    parser.add_argument("--machine-a", required=True, help="machineA daemon endpoint, host:port")
    parser.add_argument("--machine-b", required=True, help="machineB daemon endpoint, host:port")
    parser.add_argument("--machine-a-devices", default="0,1")
    parser.add_argument("--machine-b-devices", default="0,1")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--session-timeout", type=float, default=60.0)
    parser.add_argument("--session-listen-host", default="0.0.0.0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    machine_a_devices = _parse_device_ids(args.machine_a_devices)
    machine_b_devices = _parse_device_ids(args.machine_b_devices)
    chip = _build_vector_chip_callable(args.platform, args.runtime)
    digest, payload = _remote_chip_register_payload(chip, platform=args.platform, runtime=args.runtime)

    worker = Worker(level=4, num_sub_workers=0, remote_session_timeout_s=args.session_timeout)
    remote_buffers = []
    parent_keepalive = []
    try:
        worker_a = worker.add_remote_worker(
            RemoteWorkerSpec(
                endpoint=args.machine_a,
                platform=args.platform,
                runtime=args.runtime,
                device_ids=machine_a_devices,
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
                device_ids=machine_b_devices,
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

        tensor_nbytes = ELEMENTS * FLOAT_NBYTES
        group_values = {
            worker_a: (2.0, 3.0, 4.0, 5.0),
            worker_b: (6.0, 7.0, 8.0, 9.0),
        }
        group_handles = {}
        output_arrays = {}
        for worker_id, (a0_value, b0_value, a1_value, b1_value) in group_values.items():
            handles = [worker.remote_malloc(worker=worker_id, nbytes=tensor_nbytes) for _ in range(TENSOR_COUNT)]
            remote_buffers.extend(handles)
            group_handles[worker_id] = handles
            input_arrays = [
                _make_array(a0_value),
                _make_array(b0_value),
                _make_array(0.0),
                _make_array(a1_value),
                _make_array(b1_value),
                _make_array(0.0),
            ]
            for handle, array in zip(handles, input_arrays):
                worker.remote_copy_to(handle, array, tensor_nbytes)
            output_arrays[worker_id] = {
                "f0": (_make_array(0.0), _expected(a0_value, b0_value)),
                "f1": (_make_array(0.0), _expected(a1_value, b1_value)),
            }

        def parent_orch(orch, _args, cfg):
            args_a = _make_remote_group_args(group_handles[worker_a], digest)
            args_b = _make_remote_group_args(group_handles[worker_b], digest)
            parent_keepalive[:] = [args_a, args_b]
            orch.submit_next_level(remote_handle, args_a, cfg, worker=worker_a)
            orch.submit_next_level(remote_handle, args_b, cfg, worker=worker_b)

        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4
        worker.run(parent_orch, config=config)

        for worker_id, handles in group_handles.items():
            worker.remote_copy_from(handles[2], output_arrays[worker_id]["f0"][0], tensor_nbytes)
            worker.remote_copy_from(handles[5], output_arrays[worker_id]["f1"][0], tensor_nbytes)

        for worker_id, outputs in output_arrays.items():
            for name, (array, expected) in outputs.items():
                max_diff = max(abs(float(array[idx]) - expected) for idx in range(ELEMENTS))
                if max_diff > 1e-4:
                    raise AssertionError(f"worker {worker_id} {name} golden mismatch: max_diff={max_diff}")
        print(
            "remote L4 L3-group NPU smoke passed: "
            f"{args.machine_a}[devices={args.machine_a_devices}], "
            f"{args.machine_b}[devices={args.machine_b_devices}], "
            f"elements={ELEMENTS}"
        )
    finally:
        parent_keepalive.clear()
        for handle in reversed(remote_buffers):
            try:
                worker.remote_free(handle)
            except Exception:  # noqa: BLE001
                pass
        worker.close()


if __name__ == "__main__":
    main()
