# A3 Fabric Host 通信域：双机验收

本入口使用 simpler 的 L4/L3/L2 域、统一 Buffer、显式设备 view 和生产 materializer；不依赖 HIXL。Host backing 由 VMM 分配，以 Fabric handle 导出；不调用 `aclrtIpcMemGetExportKey`。

## 资源与授权

每节点由共用 A3 Host backend 分配一个 VMM/Fabric Host backing，Host 不占 NPU rank。各 chip 在 PREPARE 返回实际 BareTgid；L4 收集后先调用每个 owner 的 Fabric PID 授权，再进入 IMPORT。失败时不继续导入；owner 保留到所有 view 释放确认。远端参数传递资源身份，接收节点建立自己的映射选择，不传发送方裸地址。

顺序为 `COMM_INIT → PREPARE_EXPORT → AUTHORIZE → IMPORT → COMMIT → 使用 → VIEWS → OWNERS`。报错中的控制阶段、node/chip、ACL 操作与返回码用于定位；只有完成 COMMIT 后才允许消费。部分初始化、超时或释放未知不能被视为资源已回收。

## 两机准备

两节点都更新 `feat/a3-host-domain-multinode-clean` 并在各自环境重新安装 `pip install -e .`。当前实现是 Fabric。不要再运行旧 `examples.workers.l4.hostregister_transport.main`：该实验及 IPC 探针已删除。如仍见 `rtsIpcMemGet ExportKey`，先核对实际执行入口、两节点源码提交及 `simpler.__file__`，不要把旧 IPC 实验的失败当作本入口的 Fabric 结果。

两节点需有实际可达的 HCCS，使用相同分支构建、CANN/PTO、Python 环境及源码路径，MPI/SSH 控制面可达。先通过所在环境的设备队列获取指定设备使用权，再启动实验；以下命令不替代队列预约。

```bash
python -m tests.st.worker.comm_domain.host_domain_matrix \
  --local-host <控制节点IP> --remote-host <第二节点IP> \
  --local-devices <卡0>,<卡1> --remote-devices <卡0>,<卡1> \
  --python <两节点相同的绝对Python路径>
```

默认矩阵验证 CPU 本地 Host 读写及设备 kernel 对 Host DRAM/HBM 的双向访问、哨兵与连续交接、最终集体释放。CPU 发起的非本地访问由设备 kernel 协助，输出标为 `DEVICE_ASSISTED`，不代表 CPU 直接映射远端。初始化、传输和快照读取分属独立 Run，等待实际任务完成后再校验；`--remote-host` 省略时为单节点对照。命令中的 MPI/SSH 仅交换控制信息，不用 TCP payload-copy 冒充 HCCS。

## PTO 异步稀疏读取

同一命令增加 `--sparse`，复用单机的 `build_sparse_sdma` 和相同 kernel：

- 每节点 Host 表 8 MiB，所选行只有 64 KiB；HBM payload 只分配一行。
- 各节点的两张卡分别读取另一节点的表，行号覆盖首行、末行、重复 ID。
- PTO `SdmaTget → completion → consumer`，并提交无依赖计算；检查实际消费者结果及独立计算结果。
- `SPARSE_PASS` 逐次记录 node/chip/source/row，最后必须出现 `SPARSE_MATRIX_PASS`，域释放成功。

正确输出不证明延迟隐藏；需另采 device trace，并与同工作量串行方案比较，才能报告计算通信重叠及性能收益。本分支没有双机真机验收结果。

## 当前验收范围

- 303 项相关 UT 与 3 项生产模拟 ST 通过（单节点双卡、双节点四卡、TCP 双节点四卡）。
- A3/CANN 9.0.0 真机的单节点双卡 GlobalDomain 对照通过：18 项访问检查、7 次交接、最终域释放。
- direct L2 的 mapped 与 native-H2D＋mapped-NONE 两种消费通过；A3 双卡本地域 PTO SDMA 稀疏取行、完成衔接与消费者数值检查通过。
- 本地与全局 NoUC view 使用同一有限候选地址预留函数，避免独立域的地址冲突；失败时不发布地址。
- 真实双机 HCCS、跨机稀疏 SDMA 与性能重叠仍需在目标环境执行；单节点对照不替代这些结果。

## 公共消费接口与当前边界

- 单机 L3 使用 `host_window=HostWindowSpec(...)`，L4 使用 `host_windows={node: HostWindowSpec(...)}`；两者互斥，L3 拒绝 `host_windows`。
- `orch.copy_to_host_domain(domain, data, node=node, offset=...)` 与 `copy_from_host_domain` 支持 L4 按 owner 节点 staging，也支持 L3 借用域访问本节点 Host；单机自有域省略 `node`。复用域 copy 控制通道，每次最多 8 MiB，在任务提交前或前一 Run 完成后调用。控制面 staging 不作为 HCCS 传输性能证据。
- L4 的 `domain.host(node)` 是资源描述，不是本地映射；设备 `map_view` 仍在持有相应 attachment 的 L3 上选择。矩阵初始化和校验均走上述公共 API。
- 当前实现仍是每节点一个域内 Host backing，并非 Buffer 先行、多 shard/owner 或 #1836 的完整 attachment 轴。`backings` 是跨节点传递 canonical identity 的 wire 表，消费授权仍由现有 ImportRegistry 完成，不建立第二 registry。
- 控制面按可信节点模型运行，PID 来源是 PREPARE 返回的各 chip 身份；这不等价于对恶意控制端的身份认证。未授权 PID 的真机负向测试尚未完成。两端要求同一构建；copy 消息显式校验自身版本。

## Host 与 rank 的协议边界

rank descriptor 固定每个 NPU member 一条，native descriptor 版本保持 2。Host 的 `HostExportDescriptor` 只保存 profile、物理映射大小与 opaque handle，绑定命令中的 node / Buffer identity，不携带 rank。Python 命令使用 SDC2，交换阶段校验 Host source，IMPORT / COMMIT 之后禁止替换导出结果。

Host 使用独立 `comm_global_domain_import_host` 导入，返回地址供现有 ImportRegistry 建立消费绑定；`comm_global_domain_window` 仅接受真实 NPU rank。底层仍复用 VMM peer 引用和 VIEWS→OWNERS 回收，不要求为 Host 分配 communicator rank。现有 Host 消费 NPU 窗口的 attachment 行继续保留；Host provider / consumer 的完整公共 attachment 模型仍须按设计验收，当前不能宣称 #1836 全部交付。

A3 全局 Fabric HBM 与 Host view 均采用 NoUC，保证已有 SDMA 消费所需地址属性；非全局 Fabric 路径保持原策略。全局无 Host 场景也采用此映射属性，其硬件行为和性能需重新验收，不能从模拟器通过推断。
