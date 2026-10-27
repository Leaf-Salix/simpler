# Host 接入通信域：双机 ACL 设计

基于 [#2668](https://github.com/hw-native-sys/simpler/pull/2668)
的 `d4a77803bf1fa132b1eee8adda089fc533784c38`，复用其 Buffer/Tensor、grant 和跨 run 使用追踪。
本文描述实现契约；既有接口见[通信域文档](../comm-domain.md)。

## 1. 目标与边界

首个实现交付 A3 HCCS 双机闭环。沿用 Direct MPI：一个 MPI 作业内 rank 0 为 L4，
另两个 rank 各为一台机器的 L3 executor，每个 L3 管理两个 NPU child。
沿用现有 fork-before-MPI-init 启动次序和 supervisor，不增加启动器。
单机对照采用同一流程，仅减少为一个 L3 和两个 NPU。

| 节点 | 计算参与者 | 域内物理窗口 |
| ---- | ---------- | ------------ |
| A | Host CPU、NPU0、NPU1 | DRAM_A、HBM_A0、HBM_A1 |
| B | Host CPU、NPU0、NPU1 | DRAM_B、HBM_B0、HBM_B1 |

每节点只有一块域内 Host DRAM，两个本地及两个远端 NPU 均访问这同一 allocation。
四块 HBM 保持现有设备域窗口；域 payload 共六个物理 allocation。
MPI 负责进程启动和控制消息，ACL 负责内存和数据传输；控制消息只交换描述符、范围与结果。
含 Host 的域由 L4 复用逐节点 MPI 控制命令收集描述符并执行事务，暂不使用 NPU 域的
MPI group/allgather 快路径；无 Host 的域保留原流程。

| 发起者与目标 | 实际访问方式 | 完成条件 |
| ------------ | ------------ | -------- |
| CPU → 本节点 DRAM | owner CPU VA 上的受保护字节访问 | CPU 操作完成，满足 CPU/设备交接契约 |
| CPU → 其他 DRAM 或任意 HBM | 选本节点参与 NPU，ACL SDMA 读写 | 选定的同步 ACL copy 成功完成 |
| NPU kernel → 任意 DRAM | 本 chip 导入的 Device VA，通过 Tensor 参数消费 | kernel 所属 run 完成 |
| NPU kernel → HBM | 本地 Tensor，或现有 CommContext 的 peer window | kernel 所属 run 完成 |

“Host 进入通信域”包含 Host 内存供 NPU 访问，以及 Host 发起显式传输。
CPU 原生指针只用于本节点 owner DRAM；本提案不要求跨机 Host-target import 或远端 CPU load/store。
不引入 HIXL/ADXL 头文件、链接或引擎初始化，不交付 A2/A5 新能力、原子操作或自动故障恢复。

## 2. 最小公开接口

只扩展现有 L4 建域，首版不增加 L3 独立创建 Host 域的路径。

```python
domain = orch.get_or_allocate_domain(
    name="mixed",
    members=[(node_a, 0), (node_a, 1), (node_b, 0), (node_b, 1)],
    window_size=N,
    buffers=[CommBufferSpec("payload", "uint8", N, N)],
    host_windows={
        node_a: HostWindowSpec(numa_id=0, nbytes=N),
        node_b: HostWindowSpec(numa_id=0, nbytes=N),
    },
    retain_after_run=True,
)
host_a = domain.host(node_a)                  # 逻辑 Buffer
host_tensor = host_a.tensor([N], DataType.UINT8)
hbm = domain.member(0).buffers["payload"]    # 保留现有接口
```

`HostWindowSpec` 只有合法 NUMA id 和正整数字节数；一个节点一个 byte pool，不增加 Host 命名切片体系。
`host_windows` 为空时保持既有行为，非空时必须恰好覆盖参与域的全部 L3 节点。
按规范化节点顺序保存配置，纳入现有不可变快照；dict 插入顺序不影响复用，配置冲突在副作用前拒绝。
Host allocation 的逻辑大小与对齐后的映射大小分别记录，所有公开字节访问受逻辑大小约束。

下级通过现有 `orch.get_domain(ref)` 借用 handle。所属 L3 可获取 `domain.local_host()`，
复用 `PayloadPart.write(offset, buffer, nbytes=None)` 和 `read(offset, writable_buffer, nbytes=None)`。
薄 access backend 每次操作检查 owner、commit、generation、未 release 及范围，并持有现有 operation lease 和控制预约。
L4、非 owner L3 和失效对象拒绝访问；缓存访问对象不保活，不公开 CPU 裸指针或 memoryview。

在 L3 增加一条同步操作：

```python
orch.copy_domain(
    dst=domain.host(node_b),
    src=domain.host(node_a),
    worker=0,
    nbytes=N,
    dst_offset=0,
    src_offset=0,
)
```

`src/dst` 为同一域的逻辑窗口 Buffer；`worker` 必须是本节点参与该域的 chip。
两端完整 descriptor 必须匹配当前 grant，分别具备读/写权限；字节数为正，范围限于各自逻辑 Buffer。
既有 L3→chip 控制通道只发送池定位和范围，由 chip 解析自己的 Device VA。
CPU 读远端时传输到本机 Host pool 后 `local_host().read(...)`，写远端时反向操作。
HBM 使用已有 member Buffer，Host 使用 `host(node)`；不增加公开 stream、异步 ticket 或通用 Remote COPY 框架。
同物理池重叠范围拒绝，按 pool 与绝对偏移判断，不能只比较命名 Buffer identity。

### Host endpoint 的跨节点 attachment

L4 沿用 endpoint registry 和 `BackendResolver` 为每个参与 L3 的 Host consumer 规划设备窗口访问。
带 Host DRAM 的 A3 域可将跨节点 HBM payload 判为 `EXPLICIT_TRANSFER / REMOTE_COPY`；
该能力仅作用于当前域的参与节点，不改变默认服务对任意远端 Buffer 的拒绝。
payload 与 counter 分别判定，零字节 part 不生成 attachment；payload copy 不授予远端 counter 能力。

L3 只有在 Host owner 和全部本地 chip views 导入成功后才绑定 attachment，COMMIT 后才允许访问。
`copy_domain` 在复制本地 Host pool 与 HBM 时，校验冻结建域命令中的目标 rank、attachment、
实际 chip context 以及完整 Buffer grant；缺失 adapter、未提交、过期或被修改的 view 在发送复制前拒绝。
访问继续使用显式 `local_host().read/write` 与 `copy_domain`，不增加统一 payload 门面。
本地 `HOST_VMM_COPY`、Host-to-Host pool copy 和 NPU kernel 路径保留各自原有语义。

这覆盖现有参与节点的 Host consumer→跨节点 HBM attachment；不增加纯 CPU 节点、任意独立进程，
也不扩展通用 `RegionInstance` 分配器。对应 roadmap 4.1、4.2、4.7 的这一限定组合，不能据此关闭整项。

### 参与节点内的额外 Host 进程

配置的 `num_sub_workers` 以 `L4/L3[node]/SUB[slot]` 注册真实 Host endpoint。
L4 使用同一 registry 和 planner 生成 SUB→HBM 的 attachment，随 PREPARE 保存，
IMPORT/COMMIT 复用同一授权；未实现的 adapter 不生成 grant。

所属 L3 在控制回调首次 submit 前调用：

```python
orch.call_host(
    consume, domain.member(remote_rank).buffers["payload"], domain=domain.ref,
    staging_offset=offset, staging_nbytes=capacity, sub_worker=0, worker=0,
)
```

`consume(payload)` 在选定 SUB 进程执行，主动使用现有 `PayloadPart.read/write`。
每次请求不得超过预留 staging 容量；调用方须独占本节点 Host pool 的该区间。
SUB 经临时本地通道请求 L3，原回调线程在现有锁内完成 staging 与 ACL copy，
读结果在解锁前取出；跨节点数据继续经过参与 NPU 的现有 copy 后端。
默认权限为 READWRITE，可用 `access` 缩小权限；默认超时 30 秒、最多 64 次 IO。
临时通道使用单次 nonce，Linux 还核对实际 SUB 的 PID/UID。
调用结束撤销 payload，禁止跨调用或跨线程使用；未知 copy/control 完成状态保留资源并阻止继续使用。

这使上述限定组合包含实际额外进程的同步访问，不只是 endpoint 标签。
当前只接受同一域的精确 HBM Buffer，不支持任意进程接入、CPU-only 节点、
运行中 SUB task 的随时访问、远端 CPU 直接 load/store 或通用 Region 分配。

## 3. 内存与现有实现的接入点

### ACL 路径

L3 在既有 child fork 完成后，建立本节点参与设备的 ACL context，按需使用 native `AclRuntimeApi` 管理 Host owner：
`aclrtMemGetAllocationGranularity` 查询 `HOST_NUMA/P2P_HUGE` 粒度，
`aclrtMallocPhysical` 分配，`aclrtReserveMemAddress` 与 `aclrtMapMem` 建立 CPU owner VA，
`aclrtMemExportToShareableHandleV2(..., FABRIC, ...)` 导出同一物理 allocation。
NUMA id 由每节点配置给出，不按设备编号猜测。

每个参与 chip 使用 `aclrtMemImportFromShareableHandleV2` 导入两个 Host token，
预约 Device VA，`aclrtMapMem` 并按 CANN 约束设置本设备访问权限。
本节点 Host DRAM 同样建立 Device view，CPU owner VA 不传给 chip，也不跨进程解引用。
四块 HBM 的 allocation/export 与 peer 映射复用现有 A3 后端。
设备编号转换复用现有 logical/driver-visible 规则。

用于 SDMA 的 pool VA 必须满足 CANN NoUC 约束，包括作为端点的本地 Host 别名及 HBM。
`aclrtReserveMemAddressNoUCMemory` 要求显式地址及 1 GiB 预约粒度。
地址在 CANN 文档给定的 40–224 TiB 区间内，从现有 native 域 view 推导活跃预约区间并选择空隙；
确认释放后立即复用，不维护第二份地址注册表。外部库占用选定 VA 时明确失败，不覆盖映射、不盲目扫描重试。
这只增加 VA 预约量，不增加物理分配量。指定地址能力在 CANN 文档中仍为实验接口，需按目标 CANN/A3 环境验证。
同一组 NoUC Device view 同时服务 SDMA 与 kernel。

选定 chip 解析两端 Device VA 后固定执行 `aclrtMemcpyAsync + stream synchronize`。
CANN 对 VMM Device 间拷贝列出的接口采用此路径。
沿既有 stream 创建通路为每 chip 持有一个私有 copy stream，跨域复用；
每次 copy 必须完成同步才能返回成功。同步失败保留未知完成状态，阻止 view/owner 回收及设备 finalize。
不在 copy 出错后自动改试另一种实现；第一次调用可能已写入部分数据。
既有 stream 销毁是 best-effort，不将其返回值作为物理销毁证据；域回收依赖每次 copy 的完成同步。
新增 ACL 符号只在 Host-enabled A3 路径按需解析，保留其他后端的原初始化行为。

### 描述符与授权

Host 配置只保存 owner node、NUMA、逻辑大小和 canonical identity；HBM、Host 共用一张有序 export descriptor 表，
分别占 `[0,R)`、`[R,R+H)`，mapping size 与 Fabric token 只保存一次。`SDC3` 编码配置、backing 引用、Host 窗口与额外 consumer 授权（每个 SUB 的 endpoint 身份和逐 rank 授权集合）；旧布局明确拒绝，各参与进程须使用同版本。
Host 不占 NPU rank，不改固定 `CommContext` ABI。
每个 Host pool 只有一个 canonical identity；各 chip VA 不同，但 backing 身份相同。

公共 Host Buffer 使用无地址的 `REMOTE_SIDECAR`，不增加 backend 或修改 Buffer ABI。
域内 `[0,R)` 对应设备窗口，`[R,R+H)` 按规范化节点顺序对应 Host pool；这些槽位仅保存在私有域状态中。
Host Buffer 的 `address_space=DEVICE` 描述 NPU 的访问视图；CPU 访问通过 `local_host()`，不公开 Device VA。

Host 域的 owner L3 在 PREPARE 为实际 HBM backing 创建唯一身份，L4 汇总地址无关引用后随 IMPORT 分发，
各 owner 核对自己的原始 backing。普通 HBM 域沿用统一 Buffer 的 IMPORT/COMMIT 流程。
每 chip 在现有 ImportRegistry 中预注册自己的 Host Device 映射，缓存精确匹配完整 descriptor；
未知引用不能触发隐式映射。释放 views 前先撤销缓存，避免继续解析已释放映射。
显式 copy 使用不可变身份到槽位的授权表；peer HBM 的 kernel 访问继续通过 CommContext。
COMMIT 每个接收节点发布各 Host pool 一次；跨节点转发按接收 session 选择 locator，所有别名保持同一 canonical identity。
Host Tensor 沿用普通 TaskArgs 与跨 run 登记。
任何仅通过 Ref、闭包或 scalar 发起的后代访问，包括 CPU 字节操作、同步 copy 和 CommContext 使用，
根 L4 callback 必须在提交后代前 `get_domain(ref)` 登记本 run；L3 借用 handle 本身不为 L4 owner 保活。

配置阶段不分配 identity；创建事务分配一次 Host identity，各端按命令登记同一无地址引用，复用时不重新分配。
`_GlobalNodeDomainState` 及现有清理 journal 持有 L3 native Host owner，后者保存 handle、CPU VA、实际大小和释放进度。
chip 的 `DomainAllocation` 保存 imported views，`window_at(slot)` 从现有 view 取 VA，Python grant 与 native view 共用 slot 顺序。
显式 copy 的本地控制请求只携带域身份、两端 Buffer identity、偏移和字节数；
L2 查同一份已登记 Buffer 得出 slot、切片范围与权限，native 再按 slot 和映射范围检查，禁止请求直接提供 VA。
不另建 owner/address registry、session、launcher 或 adapter；可从既有状态推导的字段不重复缓存。
不复用会搬运 payload 的 `COPY_TO_DOMAIN/COPY_FROM_DOMAIN` staging 命令冒充 Fabric copy。

```text
建域：L4 allocate → L3 prepare owner/export → chip import → 全员 commit
CPU：L3 local_host → 现有门禁/lease → owner read/write
搬运：L3 copy_domain → 已有 mailbox → chip 校验/解析 → 同步 ACL copy
计算：TaskArgs Host Tensor → 既有 grant resolver → 本 chip Device VA → kernel
回收：既有 run fence → 所有 VIEWS 确认 → 所有 OWNERS 确认 → freed
```

## 4. 建域、排序与回收

### 建域与复用

沿用 `PREPARE_EXPORT → IMPORT → COMMIT`。每节点只创建一次 Host owner，准备现有 HBM 和 Host exports；
控制面核对节点/池完整性、大小与域身份，向所有 chip 发送完整描述符。
全员 import 成功后才发布 Buffer grant 与 CPU 接口，精确复用不重新分配、导入或清零。
发送 PREPARE 前登记 attempted 节点；响应丢失也必须纳入 ABORT，不能视作未创建。

### 完成与可见性

CPU 字节访问与同步 copy 沿用现有 direct-control 门禁：callback 内只能在本 run 首个 submit 前调用；
callback 外要求 outstanding runs 已完成，并持有现有控制预约。`scope_end` 不代替完成等待。
操作全程持有 mapping/operation lease，release 不能撤走正在使用的映射。

这些操作不创建 DAG 依赖，不自动等待已排队任务。选定实现的同步 ACL API 返回成功，
或其异步 API 后的 stream 同步成功，只证明本次传输完成；
跨节点同范围交接由已完成的全域 L4 run 或明确任务依赖排序。
CPU↔NPU 可见性采用 CANN 支持的内存属性与同步契约，须用真实同址交接验证；
普通 CPU fence 不能替代缓存一致性证明，不加入 EL0 cache 指令或 DMA 回读来掩盖 CPU 校验失败。
首版不承诺无序并发写的一致性。

### 正常释放与 ABORT

在既有 RELEASE/ABORT 命令增加内部释放阶段，Host-enabled 域执行相同顺序：

1. 撤销新使用，等待已接纳的 run、kernel 和 copy 完成。
2. `VIEWS`：所有相关节点/chip 释放 imported Device views，收齐每个必要节点的成功确认。
3. `OWNERS`：确认齐备后，各节点释放本地 Host owner 和 HBM owner；全员成功才标记域 `freed`。

Host-enabled 域拒绝绕过阶段的完整单节点 free；无 Host 的域保留现有释放入口。
每条命令带域身份、generation 和阶段，重复调用幂等；阶段进度记在既有域状态。
ABORT 覆盖全部可能创建/导入的节点，准备失败的局部清理不能提前释放已发布 token 的 owner。
不能先对 A 完整 free，再询问 B 是否仍持有 A 的 view。

view 释放失败、确认丢失或设备完成未知时，域失效并停止新访问；owner 进程存活时不主动 free，
保留清理责任并报告失败/UNKNOWN。copy 同步超时不证明 DMA 已静止，不能因此 unmap。
底层操作只有成功才撤销对应资源状态，禁止清空句柄后将失败当作可重试成功。
`Worker.close()` 复用 #2668 journal，但 Host-enabled 域仍须经过同一阶段门禁；
不能由通用 Buffer 清扫、RAII 析构或整树 teardown 提前释放尚未确认安全的 owner。
VIEWS 或 DMA 完成未确认时，正常 close 保留相关 native context，不能继续会销毁它的 device finalize/reset。
若转为作业终止，则按故障退出记录 UNKNOWN，不能报告有序释放成功。
chip finalize 回执不证明 L3 持有的 Host allocation 已释放。
borrowed handle 也不能在仅完成 VIEWS 时标记 `freed`。
MPI Abort/进程退出可能回收内存，本提案不保证失联 owner 留存或恢复；未知回收不报告成功。

## 5. 实现约束

若需要别名，逐个登记其 lease 与清理责任，不增加物理 payload 副本。
每处新状态须有唯一写入者、使用者和撤销点；测试复用既有 fixture 与参数化流程。

通用 Region 的最终规划复用与物化相同的 shape/layout 检查，单条 adapter 可用不等于整个 Region 可执行。
当前支持 root Host consumer 与 AICPU provider，以及 A3 同节点 L3 下两个不同 chip 的 AICPU provider/consumer；
payload、counter 均为正数。其余形状在资源分配前拒绝，不把实现缺失表述为硬件不支持。
额外受管 Host 的跨节点访问继续使用前述 endpoint/attachment 与域内 copy，域专用能力不提升为全局默认能力。

同节点 peer 复用现有 Region 接口：

```python
region = orch.create_worker_chip_region(
    worker_id=0, consumer_worker_id=1, payload_bytes=512, counter_bytes=128,
)
owner_binding = region.descriptor_scalars(worker_id=0)
peer_binding = region.descriptor_scalars()  # consumer chip 的地址
```

两组 binding 都沿用六 scalar ABI，作为对应 chip 的 TaskArgs scalar 前缀；Host 侧的
`payload_read/write/counter` 不适用于这个 device consumer 形状。每个 chip 校验本地完整 binding，
父级还核对当前 run 与提交目标；跨任务访问通过显式依赖或已有 counter 协议排序。
数值相同且指向同一 backing 的合法本地 binding 不因进程不同而被拒绝；裸 scalar 不是安全隔离机制。
不增加新的 Region facade，也不引入 HIXL。生产后端使用 CANN VMM，仿真使用共享 POSIX backing。

peer 的 payload/counter 都在实际消费 chip 进程中导入，设备间 peer-access 状态必须确认后才能发布 view。
Region 复用既有 run 清理：所有 view 成功撤销后才释放 provider；部分导入失败、回收失败或确认未知
保留清理记录并阻止新准入及正常 Worker teardown，后续 close 不能把遗留失败当成功。
这条同节点 VMM 路径与跨节点 Host copy 分别验收，不将仿真通过解释为硬件链路证明。
