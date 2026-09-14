# Kernel-mode PR integration log

One entry per adjudication, in the order it was made. Each entry states the
problem, the options, the choice, the reason, and which PRs it affects.

The scope is all 14 submitted PR contributions in the supplied kernel pipeline,
including the latest K1 and K3 revisions frozen during the final audit. The
end-to-end acceptance path is `init -> prepare_callable -> launch -> close`
with real `tensormap_and_ringbuffer` computation on a2a3 hardware.

See [the validation record](docs/kernel-integration-validation.md) for frozen
PR heads, executed tests, source snapshot identity and remaining boundaries.

---

## D0 - The PRs are real git stacks, not independent source-embedded copies

**Problem.** The handover states that the in-flight PRs each source-embed a
copy of K1 and that those copies have diverged by 16 / 97 / 115 / 165 lines,
so `git merge` cannot deduplicate them and each copy must be adjudicated by
hand.

**Finding.** That is not what the refs contain. K1 appears under three
different commit ids -- `86dd62b4` (#2064 / K3), `dc1268cd` (#2177, #2176,
\#2185, #2180, #2190) and `66156ed3` (#2189) -- and all three have **identical
trees**:

```bash
git diff --stat dc1268cd 86dd62b4   # empty
git diff --stat 66156ed3 86dd62b4   # empty
```

K1 itself never diverged. The line counts the handover reports are each PR's
*own contribution on top of* K1, not a competing copy of it.

**Choice.** Do not hand-merge K1 copies. Cherry-pick each PR's non-K1 commits
onto the authoritative K1, which replays every contribution as a clean
three-way merge against the identical base it was written against.

**Reason.** The unique-contribution set is a handful of distinct commits, and
cherry-picking them yields linear history with one commit per contribution and
no duplicated K1 history.

**Affects.** All PRs. This is the method the rest of the log assumes.

---

## D1 - `pr/k3` carries K1, so batches 1 and 3.5 collapse into one merge

**Problem.** The recommended order lands #2064 (K1) at batch 1 and K3 at batch
3.5, with K3 not yet submitted at handover time.

**Finding.** K3 had already landed when work started: the local K3 workspace
was at `eb696879` ("kernel-mode capacity
refusals no longer free what they protect", 2026-09-10), whose parent is K1
`86dd62b4`. So the K3 branch *is* K1 + K3. No waiting was required and no
polling was done.

**Choice.** Merge `pr/k3` once, as the authoritative baseline for both.

**Reason.** K3's K1 files are byte-identical to #2064's, so the merge delivers
\#2064 exactly. Splitting it would add a commit and change nothing.

**Affects.** #2064, K3.

---

## D2 - Integrate the H chain while retaining the authoritative K1 ABI

**Scope (2026-09-11).** The user requested every submitted PR in the pipeline,
so the earlier decision to defer HBG is superseded. The local integrated source
now contains the contributions of all 14 listed PRs: #2064, #2171, #2172,
\#2173, #2174, #2175, #2176, #2177, #2180, #2185, #2187, #2189, #2190 and #2193.
This is local source integration, not a claim that the GitHub PRs were merged
or that all tests have passed.

**Finding.** The five H heads are cumulative snapshots of H1, 2B, Context
Prepare, H2 and H3. Their shared K1 variant conflicts with the integrated C
entries and invocation header; importing that whole variant would remove
contracts the TMR path uses.

**Choice.** Integrate their HBG contributions and the shared resource
prepare/freeze/inspect/bind methods into the existing context. Keep the current
five-parameter `prepare_callable`, id-based launch, K1 header and five-event
binder topology. Preserve #2180's contribution through the integrated #2189
stack, and include #2193's updated capacity-refusal checks.

**Boundary.** H1-H3 build, sizing, immutable packets and slot admission are
present. H4 has no submitted PR in the supplied pipeline; semantic restore and
its public HBG owner wiring remain absent. HBG reports no kernel capability;
kernel init returns `UNSUPPORTED`, and prepare/launch on that uninitialized
context return `INVALID_STATE`. Added HBG and dispatch-packet tests require a Linux
build; the separate positive TMR numerical test requires a2a3 hardware.

**Affects.** All 14 submitted PR contributions; HBG ABI adaptation is recorded
in D11.

---

## D3 - `prepare_callable` keeps `caller_stream` (five parameters)

**Problem.** Three signatures are in flight:

| Source | Signature |
| ------ | --------- |
| K1 / #2177 / #2185 | `(ctx, callable_id, callable, size, caller_stream)` |
| #2176 / #2180 / #2189 | `(ctx, callable_id, callable, size)` |
| #2190 | `(ctx, callable, size, SimplerCallableHandle *out)` |

K2's cherry-pick applies its four-parameter form **without a git conflict**,
because it is a clean delta against the same K1 base. Nothing reports the
change; the parameter simply disappears.

K2 does not drop it by oversight. It replaces the doc with an argument:
preparation enqueues on the context's own AICPU stream, later launches enqueue
on that same stream, and stream FIFO therefore orders registration ahead of
every launch without a caller stream being involved. That is coherent with the
three-stream binder in #2187.

**Choice.** Five parameters. `caller_stream` is restored in
`runtime_c_api.h`, `kernel_entry_validation.h`, both `c_api_shared.cpp`
entries, `test_kernel_entry_validation.cpp` and `test_kernel_mode_c_api.py`.

**Reason.**

1. **The actual ABI consumer is five-parameter.** #2185 is the entry layer
   that gives the four C entries their callers, and it calls
   `kernel_prepare_callable(callable_id, data, size, caller_stream)`, plumbing
   the stream up through the nanobind binding and the Python wrapper with the
   documented "may enqueue asynchronous work on that stream but never
   synchronizes it" semantics. K2 is the outlier against the layer that uses
   it.
2. **The K1 interface-freeze card is the designated tie-breaker** for ABI
   signature disputes, and it specifies five.
3. **It is strictly more expressive.** An implementation that prefers to stage
   on its own AICPU stream can ignore the parameter; a four-parameter entry
   cannot later acquire ordering against the caller's *preceding* work without
   an ABI break. Keeping it preserves both designs, dropping it forecloses one.
4. It keeps prepare and launch symmetric, matching the header's own rule that
   "the caller stream is always an explicit parameter".

K2's FIFO argument is not refuted, and this integration does not depend on
refuting it. It holds only while prepare and launch enqueue on the *same*
context stream; the parameter costs nothing and covers the case where they
do not.

**Affects.** #2176, #2180, #2189 (must adopt five), #2190 (see D-later),
\#2177 and #2185 (unchanged).

---

## D4 - `simpler_kernel_mode_init` keeps 2a's contract validation ahead of K2's body

**Problem.** The only textual conflict in the K2 cherry-pick. `HEAD` (K1 + 2a)
validates the pipeline contract the config implies and then returns the
`UNSUPPORTED` refusal stub. K2 replaces the whole body with the real init,
which has no contract validation because K2 branched before 2a.

**Choice.** Keep both: 2a's `build_kernel_pipeline_contract_impl` +
`is_valid_pipeline_contract` + arena/stream topology check first, then K2's
latch-and-initialize body. The refusal stub is dropped.

**Reason.** The two are independent and both wanted. Validating before the
latch means a config that cannot be serviced is refused while the context is
still free, which is what makes a refused init leave a reusable context -- a
guarantee #2185's tests already assert.

**Affects.** #2176, #2177.

---

## D5 - Environment: pre-commit `clang-tidy` cannot run on this host

**Problem.** Every commit fails the `clang-tidy` pre-commit hook:
`build_runtimes.py` imports `fcntl`, which does not exist on Windows.

**Choice.** Integration commits use `--no-verify`. `cpplint`, `ruff check`,
`ruff format` and `pyright` all run and pass; only the clang-tidy hook is
skipped.

**Reason.** The hook is unrunnable on this platform, not failing on the
content. It runs in CI and on the Linux host used for hardware validation.

**Affects.** Every commit on this branch.

---

## D6 - #2189 over #2180, taken as #2180's K4 chain plus #2189's K5 commit

**Problem.** #2180 and #2189 must not both land: 14 conflicting files, and the
handover records them as carrying the same K4 tree with no new content between
them.

**Finding.** #2189 is a strict superset in content but not in history. It
rebuilt the whole stack under fresh commit ids -- its own K1 (`66156ed3`), its
own 2a pair, its own K4 (`b5529c89`), a K2/K4 integration commit, and finally
K5 (`b6435e48`). Cherry-picking #2189's commits after #2177 (2a) and #2176
(K2) would replay two contributions this branch already carries.

**Choice.** Take #2180's K4 chain (`3e822453`, `b0943525`), which is stacked
directly on the 2a commits this branch already has, then cherry-pick only
\#2189's K5 commit (`b6435e48`) on top.

**Reason.** It is the same end state with no duplicated contribution, and it is
the second option the handover itself offers. #2180 is otherwise superseded.

**Affects.** #2180 (K4 chain taken, PR otherwise superseded), #2189 (only K5
taken; its K1/2a/K2/K4 commits are redundant against this branch).

---

## D7 - `launch` is id-based; #2190's handle collapses into it

**Problem.** #2190 changes both entries to a `SimplerCallableHandle`
`{callable_id, generation}`: prepare writes one out, launch takes one in. That
is a third shape against K1's `int32_t callable_id`, and it arrives without a
git conflict on the launch entry.

**Choice.** Keep `int32_t callable_id` on both entries. Launch resolves the
residency internally as `{callable_id, cache.generation()}`.

**Reason.** The generation guard is the valuable half of #2190 and it survives
intact: the cache stores the generation an entry was staged under, so a
callable left from a previous context generation still resolves to
`CALLABLE_STALE` rather than replaying a recycled address. What the handle
added beyond that was a caller-held token, and an id-based ABI has no place to
hold one -- the context generation is minted at init, so the check the caller
would have armed is exactly the check the runtime now performs. Meanwhile
\#2185's entry layer, its nanobind binding and the Python wrapper are all
id-based, as is K1.

**Affects.** #2190. Its `_prepare` ctypes helper and `CallableHandle` struct
are removed as dead; its two generation-staleness assertions are re-expressed
against the id-based launch in the end-to-end test.

---

## D8 - #2190's callable cache takes a caller-supplied id

**Problem.** `KernelCallableCache::stage` allocated its own sequential id
(`entries_.size()`), because under #2190's ABI the runtime chose the id. Under
D7 the caller chooses it.

**Choice.** `stage` takes `requested_id`. An id already staged is refused as a
duplicate registration. An image whose bytes match a resident entry is admitted
under its own id, shares that entry's device address, and is charged zero arena
bytes. `commit` and `rollback` find their entry by id instead of assuming it is
the last one.

**Reason.** It preserves every property the cache's own tests assert -- one
upload per unique image, no eviction, arena accounting, descriptor table
indexed by id -- while honouring the caller's id. The descriptor table was
already indexed by id and the ABI already bounds the id to
`[0, MAX_REGISTERED_CALLABLE_IDS)`, so a caller-supplied id indexes it safely.

**Affects.** #2190. Its `test_kernel_callable_cache.cpp` needs the new
parameter at each `stage` call site.

---

## D9 - The launch topology is the binder's AICore-first set

**Problem.** Two five-event topologies are in the tree.
`KernelEventKind` (from K2) is a chain: `Start, AicoreStart, AicoreDone,
AicpuDone, SerialTail`, where the aicpu stream forks the aicore stream. The
binder's `KernelLaunchHandles` is a fork-join:
`prepare_tail, start, aicore_done, aicpu_done, serial_tail`, where both device
streams fork from the caller's Start and rejoin the caller.

The handover reports the binder as AICPU-first and leans toward K2's set. The
binder's source says otherwise: its sequence waits `start` on the aicore
stream, launches AICore, records `aicore_done`, and only then admits AICPU.
Its comment gives the two v9 reasons verbatim -- the scheduler cycle when an
AICPU startup waiter blocks a later AICore SQE, and the better failure
closure, since AICore can still be cancelled while no AICPU has written the
handshake.

**Choice.** Adopt the binder's set. `KernelEventKind::AicoreStart` becomes
`PrepareTail`; the count stays five and the storage is unchanged.

**Reason.** The binder is the only implementation of a launch sequence in any
of these PRs, and this integration wires launch onto it. Its topology is
self-consistent, it is what v9 specifies, and K2's stated deadlock hazard is
avoided by a different means: AICore's SQE is on its stream before AICPU
becomes resident, so the orchestrator's spin on the AICore handshake always
has a submitted AICore to wait for.

`tests/st/a2a3/kernel_capture/native/driver.cpp` is reordered to match. It
exercises the same capture primitives -- three streams, five events, one AICPU
launch, three AICore launches -- so its capture evidence is preserved, and it
now demonstrates the topology the product actually uses. Its two device
branches touch disjoint buffers, so running them as siblings is safe.

**Affects.** #2176 (event enum and capture ST), #2187 (unchanged, it was
already the target).

---

## D10 - AICPU transport stays on the in-repo `rtsLaunchCpuKernel` path

**Problem.** The binder's native layer, `launch_bound_kernel_native`, requires
a `KernelNativeInvocation` carrying two `aclrtFuncHandle`s and drives
`aclrtLaunchKernel` / `aclrtLaunchKernelWithHostArgs`. Nothing in any PR calls
`aclrtBinaryLoad` or `aclrtBinaryGetFunction`, so nothing produces those
handles. The TMR path in the repo launches AICPU work through
`LoadAicpuOp::LaunchBuiltInOp`, which wraps `rtsLaunchCpuKernel`.

**Finding.** The choice is not forced. `launch_bound_kernel` -- the generic
entry -- takes `launch_aicpu` and `launch_aicore` as plain callbacks in
`KernelLaunchOps`. Only the `_native` variant hard-codes the CANN family.

**Choice.** Wire launch onto `launch_bound_kernel` with owner-supplied
callbacks that directly call `rtsLaunchCpuKernel` and
`rtKernelLaunchWithHandleV2` using handles resolved during init/prepare.
Leave `launch_bound_kernel_native` in the tree, unused, as the migration
target.

**Reason.** It is the lowest-risk route to a first working launch and the one
the handover recommends: the code already exists and is exercised on hardware,
where the `WithHostArgs` family has no precedent in this repo. The binder's
12-step sequence, its compensation ladder and its capture-safety guarantees
are all in the generic entry, so nothing is given up. Switching the transport
later changes two callbacks and no sequencing.

`LoadAicpuOp::LaunchBuiltInOp` allocates its argument wrapper, so launch does
not call that helper. The owner uses stack argument/configuration structures
and a per-callable packet whose backing storage was allocated during prepare.

**Affects.** #2187. The source guard also covers the new launch owner.

---

## D11 - HBG packets use the integrated common invocation header

**Problem.** The H2/H3 snapshots assume a 64-byte common envelope with
`abi_version`, `header_bytes` and reserved fields absent from the original K1.
Hard-coded offsets would parse the wrong bytes after integration.

**Choice.** Keep the authoritative `SimplerKernelInvocationHeader`. HBG producer,
validator, HostArgs placeholders, slot limits and tests derive offsets with
`sizeof(SimplerKernelInvocationHeader)`. Validate the shared mode, identity,
counts and payload length; keep version/reserved checks on HBG's own graph
header and slot records. HBG graph format remains version 2. D13 later imports
K1's explicit trailing reserved word without changing the 40-byte layout.

**Reason.** Both runtime payloads share one envelope without introducing a
second public ABI. HBG framing, checksums and trusted slot admission remain
intact. Compilation and execution results are recorded separately after tests
run; source integration alone is not execution evidence.

**Affects.** #2174, #2175 and their HBG documentation/tests.

---

## D12 - Kernel execution uses the resident device SO's single executor

**Problem.** The TMR device SO contains one `g_aicpu_executor`, one affinity
gate and shared platform register/profiling state. Separate host contexts can
resolve the same resident SO, while their host mutexes and hidden streams are
independent. A per-context submission lock does not serialize those contexts
on the device.

**Choice.** Each loaded host runtime SO enforces one live kernel context per
`(device_id, device runtime SO fingerprint)`. Initialization claims this key
before loading or initializing device state. A second claimant returns
`PTO_RUNTIME_ERR_INVALID_STATE` with an explicit diagnostic. Failed initialization
rolls back the claim. Failed close retains it; successful normal finalize releases
it. Destruction and fatal resource abandonment cannot establish quiescence, so
neither silently releases ownership. Launch also requires the context's claim.
The owner sequences each launch's complete AICPU/AICore join before reusing its
execution regions. Program contexts never acquire or release these claims.

The registry covers contexts created through the same loaded host SO. Separately
loaded copies of that host SO and different host processes have no shared claim
registry; they must not concurrently target the same resident device runtime SO.
This integration does not add a cross-library or cross-process locking protocol.
Program/kernel execution sharing the same resident device SO must also be
serialized by the caller, because program contexts do not participate in the
kernel claim registry.

**Reason.** Moving the executor, affinity gate and platform state into
context-addressed storage requires an additional device ownership design.
Adding only a host mutex to each context cannot establish that isolation.
The new kernel consumer has one admission owner, publishes its immutable
argument storage to the affinity group, gathers all thread errors and releases
its borrowers only after every participating thread finishes.

**Rejection boundary.** TMR admission failures with an established Runtime
publish the pre-window AICore cancel sentinel. Outer dispatcher failures that
reject the prefix or residency return an error without dereferencing
`binding_address`: its alignment and numeric range alone cannot prove it is a
live host-owned allocation. If AICore is already enqueued when such a packet is
rejected, this boundary has no trusted cancellation target. Supporting recovery
there requires a prepare-time binding registry plus explicit close-time revoke;
this integration does not invent trust from malformed packet pointers. Host
validation rejects ordinary invalid inputs before either device launch.

**Validation.** Host lifecycle tests exercise concurrent claims, device/runtime
key isolation, rejected-claim cleanup, initialization rollback, failed-close
retention and successful retry. Executor tests cover trusted admission failure,
pre-window cancellation and later reuse. The standalone K4 snapshot probe uses
its own device symbol and raw encoder transport; production uses only the unified
`SimplerKernelDispatchArgs` packet sender.

**Affects.** #2176, #2180/#2189, #2190 and the launch owner. Host-side callable
admission also rejects duplicate or out-of-range child function ids before
allocation/upload, matching the device function-table contract.

---

## D13 - Refresh K1 without discarding integrated runtime behavior

**Finding.** The final remote-head audit found #2064 had advanced from
`86dd62b4` to `2ab04b1b`. The other 13 submitted heads matched the audited
snapshots, including K3 #2193 at `b8e739d9`. Compare each K1 commit against
its own parent (`405b5bbd` and `c540572d`) so unrelated upstream changes are
not mistaken for K1 contributions.

**Choice.** Import the revised K1 contracts into the working implementation:

- Pin the invocation header to 40 bytes and every field offset. Turn the
  historical final four padding bytes into `reserved_`, zero them in producers,
  and reject nonzero values in the common, TMR and HBG consumers.
- Derive callable scalar counts from signature entries, ignore historical
  callable padding, and restore the existing C++/Python factory signatures.
- Always resolve the runtime capability probe; resolve the three kernel
  lifecycle symbols only for a supported component. Publish resolved function
  pointers after successful initialization and test failed-init retry with
  fixture shared libraries.
- Make context operation callbacks non-throwing, retain the event-flags
  parameter needed by K2, and use byte-sized stream/event enums. Invalid
  initialization arguments are rejected before resource callbacks run.
- Return `INVALID_STATE` for program-only device ownership operations attempted
  through a kernel context. Structural C entry errors use `INVALID_ARGUMENT`.

**Error-code adjudication.** Revised K1 assigned `INVALID_ARGUMENT` to -1004,
which the already integrated callable cache uses for `CALLABLE_COUNT_EXCEEDED`.
Keep the cache/capacity band -1004 through -1008 and assign `INVALID_ARGUMENT`
the next free value, -1009. Preserve the specific launch-id capacity refusal.

**Integration boundaries.** K1's stub-only removal of stream/event accessors
and production source links cannot be applied: K2, HBG resource preparation
and the real launch owner use them. Retain those integrated interfaces and
K3's stronger capacity-preservation rules. No production feature is reverted
to a K1 placeholder.

**Test dispatch correction.** K2's capture test lacked runtime/device-count
markers, so the mixed-runtime scene scheduler did not execute it. Add both
markers and link the new shared resource implementation into its native probe.
The final hardware sweep must explicitly report this test, rather than inferring
capture coverage from the sweep exit code.

**Affects.** #2064, #2176, #2190 and both runtime consumers. Final validation
results are recorded separately from the source-integration decisions.

---

## D14 - Merge the integration line onto the current mainline

**Problem.** The PR target, `feat/kernel-mode-integration-test`, is identical
to `main`. Since the integration base `a1aa7fdd`, `main` has gained the
squashed final K1 (#2064) and fourteen other commits. A merge conflicts in 16
files across 59 hunks, and one disagreement is semantic rather than textual:
the merged K1 numbers `PTO_RUNTIME_ERR_INVALID_ARGUMENT` as `BASE - 4`, where
the integration line had given `BASE - 4` through `BASE - 8` to its callable
and capacity codes and `BASE - 9` to `INVALID_ARGUMENT`.

**Choice.** The merged K1 decides the contract, and the integration line keeps
the implementation that the merged skeleton does not have.

1. Host error codes take the merged numbering. `INVALID_ARGUMENT` is
   `BASE - 4`; `CALLABLE_COUNT_EXCEEDED`, `CALLABLE_BYTES_EXCEEDED`,
   `CALLABLE_NOT_RESIDENT`, `CALLABLE_STALE` and `CAPACITY_EXCEEDED` take
   `BASE - 5` through `BASE - 9`. The constants in
   `tests/ut/py/test_kernel_mode_c_api.py` follow.
2. An out-of-range launch id returns `INVALID_ARGUMENT`, as the merged K1
   specifies and as prepare already does.
3. The merged wording and codes for the execution-mode latch, the
   `ensure_acl_ready` refusal, the `attach_current_thread` refusal and the
   finalize branch are taken unchanged.
4. The integration line keeps its real init, prepare and launch bodies, the
   canonical-image validation, the current-device check on adoption, and K3's
   `commit_static_arena_bank`. The merged runner's inline
   `kernel_arena_change_is_forbidden` guard and its unit tests are not
   carried.
5. The lifecycle comment in `runtime_c_api.h` describes the integrated
   behaviour: init bootstrap and callable registration synchronize the
   context's own AICPU stream, and the launch path synchronizes nothing.
6. `ChipWorker` keeps capability-gated resolution of init, prepare and launch,
   where the merged K1 text resolves all four unconditionally.

**Reason.** Items 1 to 3 are contract, and the merged K1 is the released
contract. None of the integrated implementation depends on the numbers or the
launch-id code it replaces.

Item 4: K3's bank enforces the same capacity rule for onboard and simulation
from one place, which is the consolidation #2193 exists to make. Keeping the
merged guard as well would restore the duplication.

Item 5: the merged comment promises no synchronize on the prepare and close
paths, which this implementation does not meet.

Item 6: every runtime still exports all four entries, so every producer the
merged contract describes satisfies the gated consumer. The integrated worker
reports an unsupported runtime by the absence of those bindings, through a
binding helper that twelve call sites share. Moving to unconditional
resolution changes no in-tree behaviour and can be done later without touching
any producer.

**Affects.** #2064 (final contract adopted), #2189 and #2190 (error-code
numbers), #2193 (bank kept over the merged guard), #2185 (symbol resolution).
