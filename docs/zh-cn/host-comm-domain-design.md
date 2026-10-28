# Host 接入通信域：A3 双机 ACL backend

在 [#2668](https://github.com/hw-native-sys/simpler/pull/2668) 的设备域上增加每节点一块真实 Host DRAM。
沿用现有 Direct MPI launcher、GlobalDomain 事务、Buffer/Tensor 和 run 使用追踪；不引入 HIXL。

## 范围

两节点各有一个 CPU、两个 NPU，以及一块 DRAM、两块 HBM。
两个本地和两个远端 NPU 都访问同一个 Host allocation；Host 不占 NPU rank。

- CPU 读写本机 DRAM：owner CPU VA 上的 memcpy，字节操作返回时完成。
- CPU 访问其他 DRAM/HBM：本机参与 NPU 的 ACL SDMA，copy stream 同步成功后返回。
- NPU kernel 读写任意 DRAM：显式映射 Tensor，经 kernel/MTE 访问，所属 run 完成后可交接。
- NPU kernel 访问 HBM：原本机 Tensor 或 CommContext peer window，所属 run 完成后可交接。

不交付跨机 CPU 原生 load/store、A5 HostRegister、URMA、capture、自动故障恢复或通用 Buffer COPY。
MPI 负责启动与控制；ACL 负责内存和数据传输。

## 建域与映射

L4 在原 orchestration callback 中创建或复用域：

```python
domain = orch.get_or_allocate_domain(
    name="mixed",
    members=[(a, 0), (a, 1), (b, 0), (b, 1)],
    window_size=N,
    buffers=[CommBufferSpec("payload", "uint8", N, N)],
    host_windows={a: HostWindowSpec(0, N), b: HostWindowSpec(0, N)},
    retain_after_run=True,
)
ref = domain.ref
```

目标 L3 通过原 `orch.get_domain(ref)` 借用域，再选择本机 attachment：

```python
domain = orch.get_domain(ref)
host_tensor = domain.host(a).tensor([N], DataType.UINT8)
device_view = domain.map_view(host_tensor, domain_rank=rank)
args.add_tensor(device_view, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
```

Buffer 的物理位置仍为 HOST，两个 Tensor 保持同一 backing identity；映射不复制数据，不修改原 Tensor。
映射必须显式选择 NONE；省略 transfer 保留原 HOST/H2D 默认，因而被映射入口拒绝。
映射只在持有对应 `local_context` 的 L3 创建；不能跨 Remote 或 nested Worker 转发。
每个 chip 经现有 adapter/service 判断后，在同一 ImportRegistry 注册本端 Device VA；
解析按 `(identity, mapping)` 精确命中，不回退普通 Host binding。

## Backend 边界

L3 Host owner 通过 CANN `HOST_NUMA/P2P_HUGE` 分配物理内存，建立 CPU VA 并导出 Fabric handle。
L2 将 Host export 加在原 HBM descriptor 表之后，复用原 peer view 容器导入。
`[0,R)` 为原 HBM rank，`[R,R+H)` 为按 node 排序的 Host pool，不修改 CommContext ABI。
含 Host 的域复用逐节点 MPI 控制交换；无 Host 域保留原 collective 快路径。

核心提供两个内部 backend 操作：

- `_host_domain_bytes`：检查域引用、状态、逻辑范围与控制准入，在原 provenance 锁内读写本机 owner。
- `_copy_domain_pools`：只允许本机 Host pool 与目标域 pool 之间同步复制。
  L3 在原锁内验证域引用、actor、slot、逻辑范围及 overlap，向受信 L2 mailbox 发送 slot/offset；
  L2 检查 live/generation，native 检查实际映射范围并完成 ACL copy/synchronize。

没有任意两个 Buffer 的新公开复制接口，不提前交换全部 HBM canonical identity，
不为复制在 L2 维护第二份完整 grant 表。原 HBM identity 仍由所属 L3 在 IMPORT 创建。
Host 访问的 PayloadPart 适配、初始化/校验、CPU actor 编排和 MPI 验收矩阵位于测试模块；
backend 不依赖测试代码。

当前 A3 成功路径沿用 NoUC Device views，使用 CANN 要求的显式 VA 与 1 GiB 预约粒度。
活动预约从现有 view 推导，确认释放后可回收；外部地址冲突明确失败。
全部 HBM 是否也必须使用 NoUC 尚缺单因素对照，不能由成功矩阵反推必要性。

## 使用与回收

L4 每次提交后代前在自己的 callback 中调用 `get_domain(ref)`，沿用原 run fence 保活；
仅在 L3 获取 borrowed handle 不替代 L4 的使用登记。Host 字节操作和同步 copy 使用原控制准入，
CPU 写完后才能启动设备读取，全部设备访问完成后才能修改共享 Host 数据。
映射成功不代表数据已就绪，也不是一致性或完成 fence。

回收沿用原域流程：撤销新使用、等待 accepted work，收齐所有节点 VIEWS 确认，再释放 OWNERS。
部分导入、copy/执行完成未知或 VIEWS 失败时保留资源和上下文，不能继续 reset/free owner。
域 Buffer 是借用引用，不能用 `release_buffer` 绕过全域回收。
不承诺失联后恢复或未知结果自动转成功。

## 验收代码

`tests/st/worker/comm_domain/host_domain_matrix.py` 提供单机两卡与双机四卡同一矩阵，
分别验证 CPU memcpy、ACL SDMA、kernel/MTE，检查完整窗口、guard、同址交接与释放。
`examples/workers/l4/host_domain_mpi_direct/main.py` 保留兼容启动入口。
现场命令、日志和测试结论放本地 reports，不作为接口文档的一部分。
