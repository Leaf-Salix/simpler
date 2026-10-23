# Bundled HIXL and ACL domain window coexistence

**Date**: 2026-10-23
**Status**: integration deferred; no equivalent HIXL backend or performance result

This is an agent-facing integration record. The user-facing status is in the
[Host domain design](../zh-cn/host-comm-domain-design.md).

## Question

Can CANN's installed `hixl/hixl.h` and `libcann_hixl.so` replace the explicit
Host copy path while retaining the ACL domain's physical allocations, NoUC
device views, active kernel access, grants, and staged release?

The comparison branch starts at ACL commit
`11955e69f20c73dc23e41acc04bb24f0c490d159`. No HIXL source checkout, patched
library, private symbol, or additional payload allocation is permitted.

## API boundary

The inspected CANN 9.0 public headers expose `Initialize`, `RegisterMem`,
`Connect`, `TransferSync`, `Disconnect`, `DeregisterMem`, and `Finalize`, with
`EnableUseFabricMem`. They do not expose an API to import an existing ACL
Fabric token, retrieve HIXL's exported token, or retrieve its imported device
view. `MemHandle` is opaque and serves deregistration.

The public [RegisterMem contract](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/commlib/hixlug/llmcrv2_42_27.html)
requires both endpoints to be registered with their respective engines.
An implementation detail accepting an unregistered local DEVICE alias does
not establish a supported integration path. The linked documentation is for
9.1; the installed 9.0 header inventory and hardware results below are separate
evidence, not a claim that all SDK versions behave identically.

Historical FabricMem source retains the allocation and exports it during
registration. This explains the conflict to test; it is not a substitute for
running the installed binary. Kernel consumers still need device mappings
with the same backing as the HIXL-registered memory.

## Installed-library control

A standalone C++ control linked only the installed CANN headers and libraries.
The resolved HIXL library was checked with `dladdr`; loader output also
confirmed the CANN ACL and metadef libraries. Hardware work used an exclusive
single-device `task-submit` reservation with the executable receiving that
exact device id. Each memory/order case ran in a fresh process.

| Check                                                | HBM                         | Host NUMA/P2P_HUGE DRAM     |
| ---------------------------------------------------- | --------------------------- | --------------------------- |
| Allocate, reserve and map                            | success                     | success                     |
| First ACL Fabric export                              | success                     | success                     |
| Second export of the original handle                 | `507899`                    | `507899`                    |
| Export a handle retained from the same VA            | `507899`                    | `507899`                    |
| Import that token in the same process                | `507899`                    | success                     |
| Export the imported handle                           | not reached                 | `507899`                    |
| Original HIXL Initialize, without a preceding export | `503900`                    | `503900`                    |
| ACL-export/HIXL-register order                       | blocked before registration | blocked before registration |
| HIXL-register/ACL-export order                       | blocked before registration | blocked before registration |

HIXL initialization logs identify `GetDeviceIp` failing to open
`/etc/hccn.conf`, followed by failure to generate local communication resources.
This is an initialization/environment blocker, not a demonstrated
`RegisterMem` failure or a hardware lack of HCCS capability. The probe did not
edit system configuration, invent device IPs, or replace the installed library.
Explicit ACL cleanup returned success in every case. Collection completed,
but its successful wrapper exit is not a compatibility or transfer PASS.

The first exploratory submission did not pass its locked device into the
executable; it is excluded from the results above. The complete control was
repeated with an explicit matching device before drawing these conclusions.

## Decision and reopening conditions

The current branch has no HIXL copy implementation. The inherited acceptance
runner still uses ACL and must not be reported as a HIXL result. No timing was
collected, and no ACL-versus-HIXL speed conclusion follows from this control.

The installed API inventory and repeated-export failures leave no verified
public bridge for this exact allocation model. They do not prove that all
CANN versions lack such a bridge. Reopen the implementation after resolving
the original-library initialization prerequisite and demonstrating, on that
SDK, registration coexisting with the required kernel mappings in both
directions. Run initialization-only, registration-only, and both export orders
on fresh allocations before attempting any performance measurement.

Do not replace this requirement with private FabricMem symbols, a patched
GitCode library, an unregistered alias, or extra staging buffers. Default
HIXL/HCCL mode uses different documented allocation requirements; a separate
transfer benchmark for that mode does not validate the existing mixed domain.

Once coexistence is established, retain the identical 18/72 access matrices,
7/18 handoffs, complete-window/guard checks and ordered release. Measure
initialization separately, then compare warmed transfers with identical
sizes, actor device, source/target pools and synchronous completion. Exclude
reset, validation and control-plane round trips from native transfer timing;
report end-to-end copy latency separately. Kernel operations stay common to
both branches and are not evidence of a HIXL transfer speedup.
