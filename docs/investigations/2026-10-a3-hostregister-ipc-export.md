# A3 HostRegister IPC export qualification

Date: 2026-10-29. Historical qualification evidence; runnable delivery uses
[the Fabric matrix](../zh-cn/a3-fabric-multinode.md).

A3, Linux 5.10, CANN 9.0.0, driver 26.0.rc1: HostRegister mapping IPC export
fails with 507899; normal HBM IPC export passes the positive control.
Hcomm's `RemoteIpcRmaBufferImpl::Open` explicitly rejects HOST memory.
This result does not qualify remote Host access or rule out independent RDMA paths.

The IPC/HIXL experiment sources and commands are retained in Git history at
`31c87b688`; they are not the Fabric implementation or its acceptance entry point.
The current path allocates VMM Host backing and exports a Fabric handle rather
than exporting a HostRegister device VA via IPC.
