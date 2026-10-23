# Host 接入通信域：双机 ACL 设计

状态：设计提案，接口与测试尚未实现。基于 [#2668](https://github.com/hw-native-sys/simpler/pull/2668)
的 `1251133c07adfd25610614821a8bf80d45dc5235`，复用其 Buffer/Tensor、grant 和跨 run 使用追踪。
本文描述待交付行为；当前可用接口见[通信域文档](../comm-domain.md)。

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

以下为提议接口。只扩展现有 L4 建域，首版不增加 L3 独立创建 Host 域的路径。

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
同一组 NoUC Device view 同时服务 SDMA 与 kernel；单卡预验证已覆盖父进程持有 DRAM、子进程导入及完整窗口校验。

选定 chip 解析两端 Device VA 后固定执行 `aclrtMemcpyAsync + stream synchronize`。
CANN 对 VMM Device 间拷贝列出的接口及硬件预验证均采用此路径。
沿既有 stream 创建通路为每 chip 持有一个私有 copy stream，跨域复用；
每次 copy 必须完成同步才能返回成功。同步失败保留未知完成状态，阻止 view/owner 回收及设备 finalize。
不在 copy 出错后自动改试另一种实现；第一次调用可能已写入部分数据。
既有 stream 销毁是 best-effort，不将其返回值作为物理销毁证据；域回收依赖每次 copy 的完成同步。
新增 ACL 符号只在 Host-enabled A3 路径按需解析，保留其他后端的原初始化行为。

### 描述符与授权

Host pool 附属于现有域事务，单独携带 owner node、逻辑大小、mapping size、canonical identity 和 Fabric token。
Host 不占 NPU rank，不改固定 `CommContext` ABI。
每个 Host pool 只有一个 canonical identity；各 chip VA 不同，但 backing 身份相同。

`DOMAIN_WINDOW` body 保持 28 字节，原 `rank` 字段明确改为内存窗口槽位 `window_slot`：

```text
domain_id:u64 | generation:u64 | window_slot:u32 | slice_offset:u64
```

域有 R 个设备窗口、H 个 Host 窗口时，`[0,R)` 仍对应 device rank，
`[R,R+H)` 按规范化节点顺序对应 Host pool；NPU 成员表和 CommContext 仍只有 R 项。
不增加 kind 字段、BackendKind 或 Buffer 总布局。槽位含义由同一份不可变域元数据确定。
通用 codec 校验格式与最大槽位数；当前 H≤R≤64，最多 128 个槽位，端点再按实际 R+H 校验。
字节数不变不代表语义兼容；通信版本、绑定一致性检查和旧端拒绝仍须更新。
Host Buffer 的 `address_space=DEVICE` 描述 Tensor 的 Device 访问视图，物理归属由槽位元数据标明。

每 chip 复用 `_L2GlobalDomain.buffers` 的 identity→Buffer 映射，登记 import/commit 成功的窗口切片，
供显式 copy 核验本地/peer HBM 和 Host 两端；Tensor 解析仍额外执行设备 slot 的正确目标 chip 校验。
复用 ImportRegistry，包括缓存命中时的 grant 重验、完整 descriptor 比对、generation 和撤销检查。
保留普通 HBM Tensor 的正确目标 chip 校验；peer HBM kernel 继续用 CommContext，显式 copy 单独按域 grant 解析。
Host Tensor 沿用普通 TaskArgs 与跨 run 登记。
任何仅通过 Ref、闭包或 scalar 发起的后代访问，包括 CPU 字节操作、同步 copy 和 CommContext 使用，
根 L4 callback 必须在提交后代前 `get_domain(ref)` 登记本 run；L3 借用 handle 本身不为 L4 owner 保活。

| 接入位置 | 必要增量 | 复用内容 |
| -------- | -------- | -------- |
| `comm_domain.py`、`task_interface.py`、`orchestrator.py` | HostWindowSpec、host/local_host、copy_domain | handle、Buffer、PayloadPart、生命周期入口 |
| `global_comm_domain.py`、`worker.py` | 附属 Host 描述符、pool grant、分阶段回收 | 现有域状态、版本化 codec、PREPARE/IMPORT/COMMIT/ABORT、run 跟踪 |
| `python/bindings/task_interface.cpp` | Host owner 与受保护字节访问 | AclRuntimeApi；参考 RegionVmmAllocation 的成功后撤销记账 |
| A3 `comm_hccl.cpp`、`ChipWorker` | Device view、pool copy 与 view/owner 分离回收 | Fabric VMM helper、控制通道、stream 创建通路 |

不另建 registry、session、launcher 或 adapter 框架，不复制 #2668，也不搬入原型整套测试 harness。

### 代码分工与调用链

实现以现有函数为入口，避免再加一层通用 manager：

| 位置 | 最小职责 |
| ---- | -------- |
| `DomainConfig.capture` | 只规范化 Host 请求配置，不分配 identity 或产生设备副作用 |
| 实际 allocate 事务、`domain_buffers` | 创建时分配一次 Host identity 写入命令；各端按命令重建 slot 描述符，复用不重新分配 |
| `_GlobalNodeDomainState` | 保存本节点 Host owner 和既有域命令中的完整描述符；释放进度属于同一状态 |
| `_prepare/_import/_commit_global_domain_node` | 各阶段补 Host owner/export/import；沿原有事务发布 grant |
| `_L2GlobalDomainStore.resolve` | 共用完整 grant/范围检查；Tensor 路径额外检查设备 slot 的目标 chip，copy 路径允许已授权 peer slot |
| `DomainAllocation`、现有 VMM helper | 保存 Host imported views；内部 `window_at(slot)` 从既有 owner/imported views 取地址，不另建地址注册表 |
| `PayloadPart` 的薄 access backend | 验证活性、范围和门禁后读写 owner VA，不增加另一种 CPU buffer API |
| `Orchestrator.copy_domain` | 验证两个逻辑 Buffer，发一条本地 chip 控制命令，等待本次 ACL copy 完成 |
| `_release_global_domain_claimed/_node` | 原有 fanout 执行 VIEWS、OWNERS 两轮，正常 release 和 ABORT 共享同一 helper |

Host owner 在 L3 的 native allocation 对象中保存 handle、CPU VA、实际大小和必要的释放进度，
由 `_GlobalNodeDomainState` 及现有清理 journal 持有；不建立全局 owner-id→allocation 注册表。
导入结果回到所属 chip 的既有域状态，Python grant 与 native view 共用 slot 顺序。
import 时将全部 HBM slice Buffer 和 Host Buffer 登记到同一个 `entry.buffers` 字典，均保留完整 descriptor；
不另存 slot→VA、identity→VA 或一份仅供 copy 使用的完整命令。
显式 copy 的本地控制请求只携带域身份、两端 Buffer identity、偏移和字节数；
L2 查同一份已登记 Buffer 得出 slot、切片范围与权限，native 再按 slot 和映射范围检查，禁止请求直接提供 VA。
不为 peer copy 复制一份 Buffer 注册表，也不因 copy 可达就允许 peer HBM Tensor 提交到错误 chip。
不复用会搬运 payload 的 `COPY_TO_DOMAIN/COPY_FROM_DOMAIN` staging 命令冒充 Fabric copy。

```text
建域：L4 allocate → L3 prepare owner/export → chip import → 全员 commit
CPU：L3 local_host → 现有门禁/lease → owner read/write
搬运：L3 copy_domain → 已有 mailbox → chip 校验/解析 → 同步 ACL copy
计算：TaskArgs Host Tensor → 既有 grant resolver → 本 chip Device VA → kernel
回收：既有 run fence → 所有 VIEWS 确认 → 所有 OWNERS 确认 → freed
```

代码 review 按调用链检查，每处新状态必须有唯一写入者、使用者和撤销点；
若一个字段能从域命令或现有 Buffer 推导，就不再缓存一份。

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

## 5. 验收与实现约束

一份参数化流程覆盖单机正对照与必测双机，不为每种方向另写脚本。
双机由现有 Direct MPI supervisor 启动完整作业，域内实际 owner 为两块 DRAM、四块 HBM。

| 发起者 | READ/WRITE 目标 | 双机检查数 |
| ------ | --------------- | ---------- |
| 两个 CPU | 自己的 DRAM，CPU owner 直接访问 | 4 |
| 两个 CPU | 各自其余五个窗口，显式 ACL SDMA | 20 |
| 四个 NPU kernel | 六个窗口，kernel 主动访问 | 48 |
| 合计 | 6 个发起者 × 6 个位置 × 2 个方向 | 72 |

READ 将目标内容读到独立结果区，WRITE 将已知花纹写到目标；分离源、目标与结果范围。
作为 copy 端点的结果区在本地 Host pool 内预留，与待测区不重叠；kernel 可用独立普通 Buffer 保存结果。
guard 位于声明的逻辑窗口内、待测区两侧，不访问对齐 padding；完整窗校验包括未修改区和已知结果区。
六个 allocation 只统计域 payload owner，测试结果 Buffer 与运行时 workspace 单独计数。
每项验证完整逻辑窗口及 guard，检测越界、未完成写入和源/结果混淆，不能只检查首字节。
各节点再对同一个 DRAM 区间执行 `CPU → NPU0 → NPU1 → CPU` 交接，按完整 L4 run 分轮等待，
并让远端 NPU 访问该区间，验证所有 view 指向同一 owner。
72 项分别标注真实执行方式，本机访问不计作跨机证明，SDMA 不计作 kernel 主动访问。
最终释放与清理必须成功；缺硬件、跳过或仅有 sim 通过不记作双机 PASS。

最小回归集合：

- 参数化 UT：配置规范化与精确复用、单 owner、pool/identity/generation/范围/grant 拒绝、释放后缓存对象失效、
  纯 Ref 后代 CPU/copy 使用的根 run 登记。
- 失败注入 UT：第二 chip import 失败、PREPARE 响应丢失、VIEWS 确认丢失及 view/owner free 失败；
  校验未收齐确认不 free owner、不 finalize/reset context、不误报 `freed`、清理责任不丢失。
- 同一 E2E 流程的单机与 Direct MPI 双机矩阵、同址交接、释放顺序和域 allocation 计数。
- 原有 HBM 域、Buffer codec 和跨 run 使用回归，以及构建/链接检查无新增 HIXL/ADXL 依赖。

实现前先确认 NoUC 合法预约、同一 backing 的 SDMA/kernel 映射和 CPU 可见性；通过后接入上述域协议。
旧原型和 #2668 的 MPI group 测试只作参考，不替代这个组合验收。
若需要别名，逐个登记其 lease 与清理责任，不增加物理 payload 副本。
不预设行数上限；每项生产增量须对应分配、描述符/grant、copy 或有序释放之一，
测试复用现有 fixture 与参数化流程。无必要的新框架或大段重复代码应在 review 中删去。
