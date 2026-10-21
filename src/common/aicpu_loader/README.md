# Simpler AICPU Dispatcher SO

Source for `libsimpler_aicpu_dispatcher.so` — a transient bootstrap-only helper
loaded by CANN's preinstalled `libaicpu_extend_kernels.so`. Its only job is to
write the bundled runtime SO bytes to the main `aicpu_scheduler`'s preinstall
path under a content-fingerprint + device-id filename:

```text
/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<device_id>.so
```

The `<device_id>` suffix isolates the paired dies of one a2a3 chip (which share
the preinstall filesystem) so they never write/rename/execute one shared file —
concurrent bootstrap on a shared file corrupted the mmap'd image and faulted
`simpler_aicpu_exec` (507018 → chip fault → 507899 cascade).

The dispatcher SO itself is **never** persisted to disk and **never** dispatches
at per-task launch time. After bootstrap, the host registers the preinstall
file via `rtsBinaryLoadFromFile` (JSON load, cpuKernelMode=0) and
resolves every symbol the runtime declares it exports via
`rtsFuncGetByName` (a miss is a build/registration error, not an optional
gap); per-task launches go through `rtsLaunchCpuKernel` on the
cached `rtFuncHandle`. The main `aicpu_scheduler` owns the dlopen of the
preinstall file; the dispatcher is out of the picture once bootstrap returns.

The source is runtime-agnostic. It is built per-arch under
`build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so`, one per arch and
shared across every runtime. A single process binding multiple runtimes
therefore shares one dispatcher SO on disk. Each bootstrap acquires a shared
file lock held until the device process exits, including abnormal exit. A later
bootstrap reclaims unlocked SOs and incomplete uploads; matching bytes are
reused. A global file lock serializes publication and reclamation, protecting
the interval before the first dlopen. The runtime and standalone tools use this
same path unconditionally and require the dispatcher built from the same
checkout. Mixed-version binaries are unsupported. Leases last for the device
process lifetime, so a long-lived process retains every SO it has used; cleanup
is triggered by the next bootstrap, rather than by each runner close.

## Argument ABI boundaries

The dispatcher bootstrap uses a private `DeviceArgs` ABI. Host-side
`LoadAicpuOp::BootstrapDispatcher()` writes a `device_args_ptr` into the
bootstrap `KernelArgs` front slot at offset 40. The device-side dispatcher
reads that pointer as an extended `DeviceArgs` object:

```text
DeviceArgs + 96:  aicpu_so_bin
DeviceArgs + 104: aicpu_so_len
DeviceArgs + 112: device_id
DeviceArgs + 120: inner_so_bin
DeviceArgs + 128: inner_so_len
```

This private bootstrap structure is distinct from the platform runtime
`KernelArgs` payload used by per-task launches. Per-task AICPU launch passes
the front-less `KernelArgs` payload directly to `rtsLaunchCpuKernel` (no CANN
launch front): `runtime_args` is at offset 0, followed by profiling/logging
fields and register tables. `device_id` is not in this payload — it reaches
the device once via `InitArgs`. AICore receives only the same
front-less `KernelArgs` via a host-owned device copy.

## Maintenance upgrade and acceptance

Historical uploads have no leases. Before enabling this implementation, reserve
the entire shared device filesystem, including both dies of a paired card, and
stop all old runtime, standalone-tool and trial-build consumers. Keep admission
closed throughout cleanup and acceptance. A successful file lock or an empty
host process list does not prove that an old device consumer has exited.

1. Confirm device consumers have exited. On the device, save a complete manifest
   of regular `simpler_inner_<fp>_<device_id>.so` and trial
   `simpler_inner_v2_<fp>_<device_id>.so` files, including their `.tmp.<pid>`
   uploads. Record counts, logical bytes, allocated bytes (`st_blocks * 512`),
   and `statvfs` available blocks and inodes. Host `df` is not device evidence.
2. Use the maintainer's supported device-reset procedure and inventory again.
   If files remain, remove only the strictly matched manifest entries through
   the device maintenance interface while consumers remain stopped. Do not
   follow symlinks, recursively clear the directory, remove CANN libraries or
   replace the cache lock file. A reset success alone is not cleanup evidence.
3. Before the first new runtime upload, verify both historical namespaces and
   their temporary files have zero entries and allocated bytes, and device
   space has recovered. Reject incomplete scans or unexplained matching special
   files. Unlinking a file still held open may not recover its storage.
4. Rebuild and deploy the host runtime, dispatcher and all six tools from the
   same checkout. Save the commit, executed binary hashes and runtime
   fingerprints. Verify a cold bootstrap, the six tool smokes and the five
   Network1 cases from issue #2642. Sequentially load and execute distinct
   builds, checking that later bootstraps reclaim exited consumers' SOs without
   clearing files between runs. Multiple active readers may retain several SOs.
5. Preserve the device inventories and execution results before reopening
   admission. If the original device was already reset, report that evidence
   limit and validate known historical fixtures on an isolated device instead;
   do not claim the new runtime automatically cleaned the original files.

The maintenance inventory/cleanup entry point must work before a complete
runtime upload; the standalone query tool itself stages a runtime and cannot
serve as a full-filesystem recovery entry point. If isolation, device access or
the pre-upload inventory cannot be established, the upgrade is not accepted.
Normal bootstrap does not perform a separate legacy cleanup or reset.

## Exported entry points

Three C-style symbols are exposed; `libaicpu_extend_kernels.so::SetTileFwkKernelMap`
dlsym's all three at load time, but only DynInit does real work:

1. `StaticTileFwkBackendKernelServer`       — stub
2. `DynTileFwkBackendKernelServerInit`      — bootstrap upload (real work)
3. `DynTileFwkBackendKernelServer`          — stub

See `aicpu_dispatcher.h` for the bootstrap protocol details (extended DeviceArgs
with `inner_so_bin`/`inner_so_len`, Build-ID-derived fingerprint).

## See also

- [`docs/aicpu-kernel-launch-mechanisms.md`](../../../docs/aicpu-kernel-launch-mechanisms.md) —
  this dispatcher is the heart of "Method 2 (Path A)". That doc compares
  it against the older tar.gz method and the broken Path B
  (`KERNEL_TYPE_AICPU_CUSTOM`), and records the four failed user-space
  workarounds from issue #822 so future readers don't re-derive them.
- [`tools/cann-examples/aicpu-kernel-launch/`](../../../tools/cann-examples/aicpu-kernel-launch/) —
  standalone reference tool implementing the dispatcher bootstrap with
  the minimum possible inner kernel; copy that as a template for new
  AICPU work.
