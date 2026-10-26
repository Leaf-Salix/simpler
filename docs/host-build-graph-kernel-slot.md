# HBG execution-slot registration and admission

These internal interfaces implement the execution-slot trust root described in
[v9 design, sections 3 and 8](https://icc.gt.tc/vllm-pto?i=2#v9-design).
The public HBG kernel owner connects prepare-time registration, launch event
ordering and context-close detachment. The prepared entry attaches immutable
Host-validated images published by ordered DMA.

## Ownership and prepare

`KernelResourcePlan` reserves five logical regions within two physical allocations:
heap, runtime/SM, Definitions, A5 scheduler, and a 320-byte registry. The last
four occupy disjoint aligned slices of the packed runtime arena. The registry
adds its bytes and preceding alignment padding to the common resource contract,
but never to a graph's four mutable destination capacities or its payload.
`GraphResourceRequirements` remains a per-graph size snapshot; the plan accounts
for the additional per-context control storage.

The resource schema and slot registration use version 3. Prepare and close
retain existing allocator
accounting and rollback behavior. No third allocation or launch-time allocation
is introduced. Different graphs use one frozen set of destination addresses and
capacities. Registry storage has the same context lifetime as those destinations.

The control sequence outside capture is:

1. Build known graphs, query requirements, and aggregate capacity.
2. `plan.prepare(context, allocator_ops)` allocates the heap and packed arena.
3. `context.freeze_resources()` freezes both addresses and capacities.
4. `seal_graph_execution_slot(context, device_id, generation, runtime_binary_id,
   out)` reads the frozen resource views and produces an independent value record.
   It requires no ReadyEnqueued transition and performs no device I/O.
5. `prepare_graph_execution_slot(..., prepare_ops)` seals and calls the supplied
   registration enqueue operation on the dedicated non-hidden AICPU stream.
   The operation deep-copies the Host record before returning and must enqueue
   device initialization, registration and binding in that order.
6. The enclosing owner marks ready after all preparation tasks have been
   enqueued. Registration and every later launch share the AICPU stream, so its
   FIFO carries that ordering and no event is published for a launch to consume.
   Caller, AICPU and hidden AICore remain distinct streams.

The public owner additionally prepares image destinations and the resident
prepared execution context on that stream, then waits for initialization before
returning the context. This work stays outside capture; replay does not repeat
context setup.

`inspect_frozen_resources` only reads frozen context views in Collecting or
ReadyEnqueued state, checking device, generation and resource schema. It does not
freeze, allocate, mark ready or relax the ready-state requirement of launch.
The owner serializes sealing/enqueue with close. A callback failure is returned
unchanged; it is not evidence that any partially enqueued work has completed.
The enclosing owner supplies partial-enqueue handling and readiness ordering.

## Wire and publication

`GraphSlotRegistration` is a 128-byte POD with its own version and checksum:

| Field | Meaning |
| ----- | ------- |
| Header and flags | Frozen-capacity, serialized-execution contract |
| Device and slot generation | Context/slot affinity; generation is nonzero |
| Runtime binary identity | Immutable runtime code/ABI identity supplied by the owner |
| Maximum packet bytes | Exact bound derived from the frozen payload capacities |
| Four destination base/capacity pairs | Heap, runtime/SM, Definitions, scheduler |
| Registry base/capacity | Independent context-owned control storage |
| Checksum | Accidental-corruption detection, not authorization |

The record is derived from the context, never reconstructed from a launch packet.
All nonempty bases are aligned, lengths are overflow-checked, and all five regions
are disjoint. The runtime binary identity must be stable for the registered
runtime/ABI; it is not a callable ID, argument hash or context generation.

`GraphSlotRegistry` is a 320-byte POD aligned to a cache line. Its first line
holds the context identity and publication state; the next two lines hold the
registration. Two independent cache lines hold per-execution restore publication
and the four live region lengths. Device control code initializes only newly
allocated, exclusively
owned storage, then publishes `Empty -> Publishing -> Ready`. The complete
candidate is checked before claiming Publishing. Ready is release-published
only after copying and flushing the record. Acquisition checks state, invalidates
and revalidates the record before publishing an output value.

Ready records are immutable. Identical registration is idempotent; conflicting
registration, malformed records and mismatched registry identity fail without
replacing any bytes. Initialization refuses an actively bound registry. The
control owner must never use initialization to reset a previously live slot.

## Kernel registration path

H5 gives kernel mode its own AICPU entry and loader manifest. Program mode keeps
using `runtime_extra_aicpu_symbols`; HBG kernel mode is reported only through
`runtime_l1_extra_aicpu_symbols`. Querying or loading one manifest does not add
entries to the other, so program execution cannot accidentally resolve or invoke
the kernel registration transaction. A2/A3 and A5 report the same HBG kernel
registration symbol even though their program-mode extra symbols differ.

The AICPU entry copies the fixed 128-byte HostArgs record into aligned local
storage before validation. It rejects malformed records before following the
registry address. For a newly owned registry it performs
`initialize -> register -> bind`; for an existing matching Ready registry it
checks the full immutable record and treats an identical duplicate as success.
Initialization follows the resident binding state, not old HBM contents, because
a new context may receive storage containing a previous generation's valid bytes.
A different record for the same registry fails without replacing the published
record. While one context registry is bound, a registration for a second context
fails before writing the second registry. External quiescence and ordered detach
are required before the next context can register.

Slot registration is context-init control work on the dedicated AICPU stream. It does
not allocate device memory, build a graph, upload a graph image, synchronize a
launch, or mutate program-mode callable registration. The public kernel owner
connects detach to context close after caller-guaranteed quiescence.

The registry-binding component stores one atomic registry pointer. Binding requires
a valid Ready record and refuses a different live pointer. It stores no slot
contents, generation history, callable table or cross-context conflict state.
The prepared kernel entry separately retains its validated execution context and
callable registrations; they do not replace the registry's live identity checks.
After graph destruction and external quiescence, device control must detach that
pointer before `context.close()` releases memory. The public close owner
performs this ordered operation.

`poison_graph_execution_slot` publishes a terminal Poisoned registration phase.
Register, bind and admission reject it. Initialization still requires a new
allocation lifetime; it must not be used to reset a failed live context. The invocation owner serializes poisoning with registration,
restore and retirement; it also propagates the native error to the Host context.
Poisoning neither drains device work nor frees resources or resets the device.

## Invocation admission

The Host canonical packet uses version 6: its 200-byte HBG header carries the
device, runtime binary, slot generation, live heap length and region bindings.
The common K1 envelope retains its ABI. `make_graph_prepared_dispatch` validates
packet framing/checksum, then runtime layout, null Host pointers, task/tensor
contents, relative argument pools, dependencies, heap ranges and Definition
structure. Rejection preserves the output descriptor and all working memory.
Graph materialization retains its topology and tensor-source semantic checks.

The Host retains an immutable pinned image and orders H2D before the fixed
432-byte `GraphPreparedDispatch` on the same AICPU stream. The entry accepts
only this prepared transport. The descriptor checksum covers metadata, not the
separately transferred image; execution relies on full Host validation, private
immutable sources and stream ordering. Every replay transfers all live regions.

Each execution role gets its participants and Runtime from independently bound
context state, then:

1. Acquires the live registry, checking Ready state, checksum, device, runtime
   binary and generation.
2. Rejects HostArgs overlap with the registry or working regions, then checks
   its fixed descriptor's version, reserved fields, size, checksum, callable
   identity and region bounds.
3. Compares every destination base/capacity pair with the independent slot.
   A recomputed checksum cannot authorize foreign destinations.
4. Compares its descriptor with every role's record. Agreement permits image
   maintenance and leader attach; disagreement collectively cancels execution.

## Shared callable admission

Callable registration and execution slots are independent authorities. The
prepared invocation's callable ID, effective tensor/scalar counts, hashes and
Runtime address must match its registration. Graph slot generation, device,
runtime binary and destination ranges must match the context registry. An
already registered callable/context generation accepts only an identical
registration. Neither authority replaces context/callable lifetime protection.

The shared `PreparedInvocationView` and `validate_invocation_header` continue
to serve other kernel submission paths; HBG's prepared entry validates its own
`GraphPreparedDispatch`.

## Leader restore and execution publication

### Host publication and prepared attach

The Host validates each new template and retains a private immutable pinned
source while DMA or a callable cache needs it. Any capture use pins the source
until externally quiescent context close, including capture that reuses an
eager cache entry. An eager template miss reclaims uncaptured sources only
after an earlier successful eager tail completes and no callable cache holds
them. A capture-only tail record is not a completion proof.

`attach_prepared_graph_image(prepared, slot, out)` receives an independently
acquired slot. Every role has maintained the DMA image and joined before this
call. Attach performs no image copy, payload/task scan or image checksum:

1. Checks metadata/slot identity and canonical runtime/SM layout.
2. Claims only an Idle attempt and advances its device-owned counter.
   Ready/Restoring return Busy, Failed returns Quarantined, and exhaustion
   returns Exhausted without wrapping the counter.
3. Wires RuntimeContext/SM/scheduler and initializes queue headers/sequences.
   Only an immediately preceding successful retirement at the same Runtime
   address permits mailbox ticket-storage reuse; all other cases initialize
   the complete mailbox.
4. Publishes control metadata, commits the successful generation and
   release-publishes Ready. Returns RuntimeContext, live SM bytes and task
   count. Failure preserves the output; an already claimed failure stays Failed.

Restore neither clears nor maintains the tensor heap. Task writers and Graph
materialization initialize data and execution metadata respectively. The Runtime
image starts at `runtime_offset`, omitting device-initialized scheduler/queue/
mailbox storage. Definition and optional A5 scheduler images have zero
destination offsets. Frozen physical arena sizes and offsets do not change
with logical queue capacities.

The separate 128-byte `GraphRestoreControl` lives in the registry allocation,
never the DMA image. Prepare still performs two physical allocations; attach
allocates none. Host construction owns vector/pinned storage and is not an
entirely allocation-free submission path. Host/device builds must match and
old captures must be recreated.

### Completion and retirement

The leader publishes this invocation's READY/CANCEL through its epoch/status
barrier. Peers consume that result rather than sampling an unversioned old
Ready flag. All execution/status readers, executor cleanup and AICore work
finish before each role maintains mutable task-state image spans. All image
maintenance finishes before the final role retires. The executor maintains
Runtime state: A3 kernel mode maintains its control region, while program mode
maintains the complete Runtime.

`retire_graph_restore(registry, attempt, completion)` checks the exact attempt,
outcome and independent runtime/teardown errors:

| Outcome | Required restore state | Result |
| ------- | ---------------------- | ------ |
| Completed | Ready, matching successful committed attempt | Idle |
| ControlledFailure | Failed, known controlled fault and clean teardown | Idle, full initialization next time |
| FatalFailure | Ready or Failed | Terminal Poisoned, no retry |

The public prepared entry submits only Completed or FatalFailure.
ControlledFailure is the lower-level controlled-cleanup contract, not an
automatic retry path for public kernel invocations. Either nonzero
runtime_status or unexpected_teardown_status forces poisoning regardless of
the claimed outcome. An entry failure before claiming an attempt poisons the
registry directly. Retirement does not clean up, free memory, reset devices or
advance committed_generation; the Host owner still propagates failure and
protects resource lifetime.

Unknown outcomes, stale attempts, duplicate retirement, success retirement of
a Failed slot and reuse after poisoning fail closed. A prior successful result
cannot authorize a new execution; the execution lease excludes concurrent reuse.

Unit coverage separately exercises Host image-semantic rejection, prepared
metadata/independent slots, ordered DMA followed by attach, queue layout and
capacity, mailbox wrap/reuse, empty graphs and retirement error classification.
Host rejection preserves output metadata and working memory. These tests use
Host-addressable memory and the real validation/attach/retire implementations;
they do not demonstrate CANN capture or device cache coherence, which require
kernel-entry/binder hardware regressions.
