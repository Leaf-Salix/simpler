# Borrowed HBG native capture and replay

## Scope and execution ownership

A2/A3 onboard HBG kernel mode borrows the caller's device, RTS context, stream
and DEVICE tensor storage. Program and kernel initialization retain different
ownership contracts, while callable registration, native preparation, submission,
completion and cleanup share the existing runtime implementation. Program graph-task
capture adaptation is outside this change. TMR, A5 and simulation capture remain
unsupported; their ordinary eager execution is not replaced.

The dedicated HBG packet executor, device slots, graph-token leases and submission
binder have been removed together with their C ABI and Worker/Python entry points.
Callers register through the common callable registry and execute through native
prepare/launch/completion. No rejecting compatibility stubs remain.

## Host/device sequence

```mermaid
sequenceDiagram
    participant U as Caller / framework
    participant H as Common native Host runtime
    participant R as RTS streams / graph
    participant C as AICPU native executor
    participant D as AICore
    U->>H: Borrowed init + common callable registration
    U->>H: Native eager prepare / launch / wait / finalize
    H->>R: Existing eager publication and submission
    R->>C: Native Runtime + graph definition
    C->>D: Existing mailbox scheduling
    D-->>C: Task completion
    U->>H: reserve_capture_resources (outside capture)
    H->>H: Reserve descriptor, pinned image, workspace, streams/events
    H->>H: Reclaim obsolete workspace outside capture
    U->>R: CaptureBegin(caller stream)
    U->>H: Common prepare_run + launch_run_on_stream
    H->>H: Validate current bindings and reserved capacity
    H->>R: Record Runtime/image restoration + replay gate + native launches
    Note over H,R: Caller/AICPU/AICore event ordering uses existing execution streams
    H-->>U: Recorded receipt (not completion)
    U->>H: finalize_run
    H->>H: Attach capture resource owners; no device wait/free/unmap
    U->>R: CaptureEnd
    loop Serial replay on the capture stream
        U->>R: Graph replay
        R->>C: Gate: epoch, sticky error, participant admission
        R->>C: Restore Runtime/image; native executor entry
        C->>D: Existing mailbox scheduling
        D-->>C: Task completion
        C-->>R: Terminal status and participant retirement
    end
    U->>R: Destroy graph after replay completion
    R-->>H: Model destruction notification
    U->>H: retire_capture_resources
    H->>R: Drain retained streams
    H->>H: Release graph-held resources after drain proof
```

RTS records stream operations. The actual task scheduler remains Simpler's native
AICPU executor and mailbox communication with AICore. Capture does not introduce
a second task executor. Ordinary eager phase order is preserved.

## Cold preparation and capture restrictions

A completed and finalized eager warmup establishes the native resource demand.
`reserve_capture_resources` reserves remaining descriptor/pinned publication
storage, execution streams and events outside capture, and reclaims obsolete
workspace before marking the reservation ready. Subsequent eager preparation
invalidates readiness; callers reserve again before the next capture.

Capture refuses missing resources or insufficient capacity rather than allocating,
replacing backing, or reclaiming obsolete storage. Both recorded-run finalize and
prepared-but-unlaunched capture finalize defer obsolete workspace reclamation.
The retained full Runtime/image is restored on every replay. The separate 64-byte
NativeCaptureControl tail is not restored: its epoch and sticky error must survive
replay. Simpler retains its descriptor, publication storage, workspace, code and
events; the caller must retain borrowed context, stream and tensor storage.

Same-stream serial replay is supported. Cross-stream replay, concurrent replay,
TMR capture, diagnostics during capture and program graph-task adaptation are not
claimed. A Recorded receipt (-1008) must not be interpreted as completed output.

The reservation names one pipeline slot and arena bank, and a recorded run must
take that same slot: a direct run that lands on another slot has no reserved
descriptor, events or publication capacity and is refused during preparation.
The public `Worker` reserves slot 0 / bank 0, so it supports **one captured run
at a time** — finalize each recorded run before submitting the next one. The C
ABI accepts an explicit slot/bank for callers that warm and select their own.

The RTS event work mode is process-wide and has one value. Kernel-context
initialization raises it from software events (0) to hardware events (1) when
the runtime supports capture, and it is not restored at close. Program-mode
contexts in the same process therefore share mode 1, and a context that already
observes mode 1 changes nothing.

A recording that fails after its first queued operation leaves an incomplete
model. It is not retried into that model: the caller abandons capture, destroys
the model, retires the retained resources and reserves again before recording a
new one.

Replay status is not host-visible. A replay the gate refuses does not advance
the epoch; the first AICPU failure is retained in `sticky_error` and refuses
every later replay. Recorded runs are already host-finalized, so poll and wait
carry no replay result: the caller's stream is the only per-replay completion
signal, and device fault notices matched to the context are consumed by a later
host-supervised finalize or close. AICore reads the epoch alone, so a replay
refused for a sticky failure still reaches AICore work with the previous epoch.
Per-replay status is a C4 item.

## Common native capture resource attachment

Finalizing the run transfers resource ownership to the existing captured-graph
attachment. It neither waits for an unplayed graph nor releases the resources
referenced by recorded commands. Destruction notification alone is not a completion
proof: explicit retirement also drains retained streams. Retirement is
context-wide: `retire_capture_resources` requires every retained model destroyed
before it runs, so a dead graph cannot be released while another captured model
is still live. Failed capture end, never-replayed graphs and failed drain follow
the same ownership rules. Failed free keeps the allocation in the existing ledger
and fails that reservation. A later reservation may proceed with retained
accounting; it does not retry an unconfirmed free or pretend the allocation was
released.

## Validation responsibilities

| Boundary / owner | Validated fact and timing | Regression coverage |
| ---------------- | ------------------------- | ------------------- |
| Common callable registry | Canonical image bounds before upload; ID occupancy and code ownership | `test_kernel_mode_c_api.py`, native admission C++ tests |
| Native prepare / invocation validation | Current borrowed context, DEVICE tensors, geometry and config before binding; generation belongs to the actual registry/context | Native admission, kernel caller-stream tests |
| Native HBG builder and bind ledger | Graph definition sizing, overflow, offsets and native resource requirements | HBG bind-ledger and definition-packing C++ tests |
| Cold reservation / capture prepare | Existing backing capacity and ready streams/events; capture may not grow resources | Native admission reservation/growth/refusal tests; cold probe |
| Common runtime publication | Stable full Runtime/image backing through asynchronous consumption; graph attachment pins it beyond finalize | Runtime launch-metadata and captured-resource tests; real replay |
| Device replay gate / terminal executor | Nonzero epoch, unfinished participants and sticky failure; terminal error survives accumulator reset | Terminal-retirement C++ tests; real repeated replay (not exhaustive device fault injection) |
| Common finalize / capture retirement | Recorded is not completion; model destruction plus drain before release; capture finalize defers reclamation | Captured-resource and native admission tests; never-replayed/failed-end board cases |

The obsolete framing facade had no production callers and
was removed with its private test suite. TMR retains its actual
`consume_kernel_invocation` framing, registered generation, runtime-address/epoch
and tensor checks; HBG uses native launch metadata and GraphDefinition validation.
The old whole-image `validate_images` replay scan is gone with the second packet
format. Native preparation validates the privately constructed image before
publication; graph retention pins its immutable replay source. Device launch
metadata and runtime access checks remain on the actual consumption path.

These checks establish different facts at ownership boundaries; there is no second
whole-invocation packet parser or fixed-slot validation pipeline. Gate arguments
are privately constructed by the Host runtime, not a new public packet ABI.

## Test migration and evidence

Old packet parser, slot-control, dedicated launch binder and fixed-slot close tests
were removed with their production implementation. They are not counted as passing
native tests. Shared native admission, publication, graph attachment and completion
tests cover the current owners. `test_native_capture_chain.py` exercises the real
C ABI, ChipWorker and Worker capture paths.

The cold probe refuses device/pinned allocations and frees, stream/event creation,
and HAL unmap while recording. A CPU self-test proves each interception, including
`halHostUnregister` obtained through an explicit local library handle. Board checks
then require zero observed cold operations across real capture and finalize.

Batch evidence and exact validation results are stored under
`build/review-validation/native-capture-convergence/`; final API-removal evidence
is under `build/review-validation/native-capture-api-cleanup/`. CPU tests do not qualify
hardware behavior; A2/A3 board results do not qualify A5 capture or arbitrary device
fault recovery.
