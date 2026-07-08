# A3 HCCL Smoke

This directory is a standalone A3 HCCS/HCCL connectivity smoke. It does not
depend on Simpler.

It checks the first hardware layer we need before integrating L4:

```text
2 hosts x 16 devices
  -> rank table
  -> HcclCommInitClusterInfo
  -> HcclAllGather
  -> verify gathered int32 payload
```

## Files

- `gen_ranktable.py`: generates `ranktable_a3_2host_32rank.json` for
  `120.9.10.37` and `120.9.10.35`.
- `hostfile.mpi`: MPI hostfile with 16 ranks on each host.
- `hccl_allgather_smoke.cc`: one process per device, HCCL AllGather smoke.
- `Makefile`: builds the smoke binary on the A3 server.
- `run_2host_32rank.sh`: launches the 32-rank smoke.

## Build

Run on a server with CANN, MPI, and HCCL available:

```bash
cd tools/a3_hccl_smoke
python3 gen_ranktable.py
make
```

`make` uses `mpic++` by default. If `mpic++` is not on `PATH`, load or
install the MPI development environment first, then run `make` again.

If your CANN package links ACL through `libascendcl.so` instead of
`libacl_rt.so`, build with:

```bash
make clean
make ACL_LIB=ascendcl
```

## Run

Copy this directory to the same path on both `120.9.10.37` and
`120.9.10.35`, then run from one host:

```bash
cd tools/a3_hccl_smoke
bash run_2host_32rank.sh
```

Equivalent explicit command:

```bash
LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH \
mpirun --hostfile hostfile.mpi -np 32 \
  ./hccl_allgather_smoke ranktable_a3_2host_32rank.json 16
```

Expected success signal:

```text
A3 HCCL AllGather smoke PASS
```

## Hang Triage

If `bash run_2host_32rank.sh` hangs, first check where it stops.

The script prints the local host, `mpirun` path, `mpirun` version, rank table,
hostfile, and launch command before starting the HCCL binary. The binary prints
progress before and after `MPI_Init`, `aclInit`, `aclrtSetDevice`,
`HcclCommInitClusterInfo`, `HcclAllGather`, and stream synchronization.

Useful split checks:

```bash
# MPI launcher only. This should print both hosts quickly.
mpirun --hostfile hostfile.mpi -np 2 hostname

# Verify remote ranks start in the expected directory and see the inputs.
mpirun --hostfile hostfile.mpi -np 2 \
  bash -lc 'hostname; pwd; test -x ./hccl_allgather_smoke; test -f ranktable_a3_2host_32rank.json'
```

Interpretation:

- No `[a3-smoke] launching mpirun`: the shell script did not reach `mpirun`.
- No `[pre-mpi] MPI_Init begin`: `mpirun` did not start the rank process.
- Stops at `MPI_Init begin`: MPI launch or SSH/rendezvous is stuck.
- Stops at `HcclCommInitClusterInfo begin`: rank table, HCCS IP/port, or
  cross-host device connectivity is the first suspect.
- Stops at `aclrtSynchronizeStream begin`: HCCL launched but the collective did
  not complete.

## Notes

- `rank_id` is global: `120.9.10.37` owns ranks `0..15`, and
  `120.9.10.35` owns ranks `16..31`.
- The program maps local MPI rank to local device id, so each host should run
  exactly 16 local ranks.
- `run_2host_32rank.sh` uses OpenMPI-compatible `--hostfile`. If it runs as
  Linux `root` and the local `mpirun` supports the option, it also adds
  `--allow-run-as-root`.
- `super_device_id` is generated as the global `rank_id`. If the server
  requires physical SDID values, replace it with the values reported by
  `npu-smi info -t spod-info`.
- `device_port` and `host_port` are generated as distinct per-device ports to
  avoid single-host port conflicts during the smoke.
