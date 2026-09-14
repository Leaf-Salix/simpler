# Kernel 模式 callable 缓存调用链

本文说明当前集成代码的 callable 准备、驻留、执行和释放流程。
公开接口以 [runtime_c_api.h](../../src/common/worker/runtime_c_api.h) 为准。

## 1. 支持范围与调用顺序

| 构件 | Kernel 模式状态 |
| ---- | --------------- |
| a2a3 / a5 onboard 的 tensormap_and_ringbuffer（TMR） | 支持 init、prepare 和异步 launch，能力查询返回 1 |
| host_build_graph（HBG） | H1-H3 内部能力已集成；H4 尚未提交，公开 init 返回 `UNSUPPORTED`，能力查询返回 0 |
| sim | 公开 kernel init 返回 `UNSUPPORTED` |

HBG 和 sim 的不支持判定发生在取得 kernel 身份之前。在这种未取得身份的 context 上，
结构合法的 prepare / launch 返回 `INVALID_STATE`。能力查询不替代 context 初始化或生命周期检查。

调用者先完成 ACL 初始化，并让当前线程持有所需设备。Kernel init 借用这个设备，
不调用选卡、设备重置或 ACL 初始化/终止接口。init 可在 capture 之外同步自己的
AICPU stream 完成启动；prepare 和 launch 不做 stream/device 同步。
调用者负责串行执行同一 context 的 init、prepare、launch 和 finalize。

```cpp
int simpler_kernel_mode_prepare_callable(
    DeviceContextHandle ctx, int32_t callable_id, const void *callable,
    size_t callable_size, void *caller_stream);

int simpler_kernel_mode_launch(
    DeviceContextHandle ctx, int32_t callable_id, const void *args, void *caller_stream);
```

`callable` 是未修补设备地址的完整 `ChipCallable` 镜像，包括 header、orchestration SO
和子 `CoreCallable`。`callable_size` 必须等于实际大小。prepare 不接收 tensor 实参，
也不执行算子。调用者指定 `[0, MAX_REGISTERED_CALLABLE_IDS)` 内的 ID；当前上限为 64。

```cpp
// 已完成 caller 的 ACL 初始化、选卡以及 simpler_kernel_mode_init。
int rc = simpler_kernel_mode_prepare_callable(ctx, 7, callable, size, caller_stream);
if (rc != 0) return rc;
// 在 capture 之外完成 caller 的 warmup，并同步 caller_stream，检查异步准备结果。
rc = caller_warmup_and_synchronize(caller_stream); // 调用者自己的逻辑
if (rc != 0) return rc;
return simpler_kernel_mode_launch(ctx, 7, args, caller_stream);
```

同 ID 再次 prepare 是重复注册，会被拒绝。不同 ID 使用相同内容时，各有驻留描述符，
但共用一次代码上传，不重复占用代码预算。成功驻留的 ID 在 close 前不删除、不换出、不复用。
公开 C API 不返回 `SimplerCallableHandle`；该类型只在内部缓存中组合 ID 和 generation。

init 接收调用者提供的非零、进程内唯一 `context_generation`。Host launch 从当前
context 查出驻留信息，将 generation 写入设备包。仅凭整数 ID 不能判断它是否来自另一
context；设备端的 generation 检查用于拒绝旧调用包，不能把旧 ID 自动变成跨 context 句柄。

init、prepare、launch、finalize 只读核对当前线程的设备身份，不替调用者选卡。
prepare / launch / finalize 查询设备失败时原样返回查询错误；设备不匹配时返回
`INVALID_STATE`。这些提前拒绝不改变驻留内容，也不把 context 标为 Poisoned；
调用者恢复正确设备后可以继续使用。

## 2. 缓存准入、去重和上传

每个 `DeviceRunnerBase` 拥有一份
[KernelCallableCache](../../src/common/platform/include/host/kernel_callable_cache.h)。
`stage` 校验 generation、ID、镜像大小、signature、名称、子项布局和子函数 ID，
拒绝已占用 ID 或未完成准备的条目，然后检查内容是否可共享以及数量/字节预算。
只有 hash、长度和完整字节都相同才共享代码；hash 相同但内容不同会被拒绝。

| 情况 | 行为 |
| ---- | ---- |
| 已占用 ID | `INVALID_STATE`，保留原条目 |
| 新 ID、相同内容 | 共用代码地址，上传该 ID 的描述符，继续其 runtime 注册 |
| 新 ID、新内容 | 使用固定代码区，修补私有副本并上传 |
| 数量或字节预算超限 | 拒绝候选，保留已有 ready 条目 |
| 存在未 ready 的条目 | 拒绝后续 stage，防止暴露未完成准备的资源 |

代码预算为 512 MiB；每份唯一镜像按 `align_up(callable_size, 64)` 计费，包含整个
`ChipCallable`。首次上传分配固定代码区和描述符前缀，后续不扩容：

```text
arena_（底层分配由 MemoryAllocator 持有）
├── 2 KiB：64 × KernelCallableDeviceResidency（每项 32 字节）
│   └── descriptor_address = arena_ + callable_id × 32
└── 512 MiB：按 64 字节对齐追加的唯一代码镜像
    └── device_address = arena_ + 2048 + 当前 used_
```

上传只修补临时 `scratch` 中的子 `CoreCallable::resolved_addr_`，调用者镜像保持不变。
当前缓存上传仍通过 `Ops.copy → rtMemcpy(..., RT_MEMCPY_HOST_TO_DEVICE)` 同步复制代码和
描述符；这不同于后续 runtime 注册的异步执行，也不意味着 prepare 会同步 stream/device。

Host 条目保存驻留信息、内容 hash、镜像副本、计费字节数和 `ready`。
`resident_count()` 只统计 ready 项，`resident_bytes()` 包括未 commit 的计费占用。
不同 ID 共享代码时仍保存各自的 Host 条目；`host_bytes()` 是去重代码内容的计量，
不是这些容器和镜像副本的实际进程内存总量。

## 3. Runtime 注册和准备完成的顺序

stage 成功后，`record_callable_on_runner` 生成 orchestration 信息和
子 `func_id → 设备代码地址` 映射。Kernel 路径从缓存取得已上传地址，不重复上传代码；
program 模式继续使用原有上传与引用计数路径。

TMR 的 `prepare_kernel_callable` 首次配置固定 runtime 区域、准备 `PersistentKernelArgs`，
随后冻结配置。每个 callable 在此分配 Host dispatch packet 缓冲区，launch 只重写内容。

设备注册在 context 专用的 AICPU stream 上发射 `RegisterCallableName`。
提交成功后记录 `PrepareTail`，并让本次传入的 `caller_stream` 等待它。
调用者同步 caller stream 即可观察注册完成或异步错误；prepare 本身不执行
`aclrtSynchronizeStream*` 或 device synchronize。

context 随后转为 `ReadyEnqueued`，缓存通过 `commit(callable_id)` 发布 ready 条目。
ready 表示准备已提交并建立依赖，设备工作仍可能在执行。第一次 launch 消费 `PrepareTail`；
调用者仍须在 capture 之前完成自己的 warmup 和同步检查。

HBG 内部准备包含资源计划、freeze 和 execution-slot 注册；公开 HBG init 已提前拒绝，
不能通过公开 prepare 绕过 H4 缺失的限制。详见
[HBG 资源契约](../host-build-graph-kernel-contract.md)和
[HBG 槽位准入](../host-build-graph-kernel-slot.md)。

## 4. Launch 和设备端检查

当前 TMR 的 Host 链路为：

```text
simpler_kernel_mode_launch(ctx, callable_id, args, caller_stream)
  ├─ 校验参数、kernel 身份和 context 状态
  ├─ 取得提交锁，核对设备及 context 的执行占用权
  ├─ cache.resolve({callable_id, 当前 context generation}, residency)
  ├─ 将 tensor 元数据、设备地址和 scalar 编入已分配的 dispatch packet
  └─ launch_bound_kernel：caller 分叉到专用 AICPU 和隐藏 AICore，再汇合到 caller
```

`args` 指向真实 `ChipStorageTaskArgs`，tensor 必须声明为设备地址空间。Host 只读取元数据，
不解引用或上传 tensor 内容。每次 launch 编码当前实参；CANN 接管调用包快照后，
调用者可复用 Host 参数对象，tensor 和 context 资源仍须保持到设备工作完成。

binder 使用三条 stream 和五个 event，先提交 AICore，再提交 AICPU，避免启动相互等待。
launch 不分配设备内存、不创建 stream/event、不同步、不查询 capture 状态。
返回 0 表示提交成功，最终数值和异步错误须由调用者同步后检查。

设备包是 [SimplerKernelDispatchArgs](../../src/common/task_interface/kernel_dispatch_args.h)
加 runtime payload。前缀包含包长、驻留描述符地址、context 的 `KernelArgs` 地址和 generation、
SM/arena 范围及 `SimplerKernelInvocationHeader`。公共 invocation header 固定为 40 字节；
`host_copy_tensor_count` 和显式 `reserved_` 必须为零。不要使用旧的 64 字节公共 header 假设。

设备入口先检查公共 framing 和驻留描述符，再由 TMR consumer 校验绑定、大小、参数数量和
signature，解码到本次调用的私有参数，进入真实 executor。旧包的 generation 不会被刷新。
HBG payload 尚未接入执行 consumer。

部分已提交工作的失败会使 context 进入 Poisoned。若外围包在建立可信绑定前被拒绝，
设备入口不会解引用任意 `binding_address` 尝试取消 AICore。常规错误输入在 Host 提交前
就被拒绝；损坏设备包的恢复不等同于已完成端到端支持。

## 5. 回滚、释放和验证范围

stage 的分配或复制失败撤销候选，已分配的固定 arena 可保留供重试。
Host runtime 记录失败会回滚未发布条目。注册开始后发生错误时，context 保留相关地址并
进入 Poisoned，调用者须建立静止状态后显式关闭。kernel 模式拒绝通过 program 注册/注销
接口绕过这些规则。

close 前，调用者停止提交、等待 eager/replay 完成并销毁引用资源的 graph。
`finalize_device` 只释放 context 资源，不释放 caller tensor，不重置设备，不终止 ACL。
设备查询拒绝或清理失败后可以重试；清理失败保留剩余资源和执行占用权。
同一加载的 host runtime SO 限制一个设备/runtime 身份下的 live kernel context；
不同 host SO 副本或进程间的执行隔离仍需调用者协调。

相关验证包括：

- [缓存单元测试](../../tests/ut/cpp/common/test_kernel_callable_cache.cpp)：ID、容量、去重、上传失败和回滚。
- [Dispatch packet 测试](../../tests/ut/cpp/common/test_kernel_dispatch_packet.cpp)：真实解码、重复编码、地址和 scalar 隔离。
- [设备入口测试](../../tests/ut/cpp/common/test_kernel_dispatch.cpp)：公共包校验、驻留身份和 generation 拒绝。
- [HBG 内部测试](../../tests/ut/cpp/common/test_hbg_host_graph_build.cpp)：构图、资源容量、包和槽位准入。
- [C ABI 测试](../../tests/ut/py/test_kernel_mode_c_api.py)：真实准备/关闭、TMR eager 数值、参数快照和设备查询恢复。

a2a3 eager 数值路径已有真机通过记录。A5 的实现与测试入口均存在，但测试定义不等于真机验收。
内部 HBG 单元测试以及 eager 成功也不代表完整 ACLGraph capture/replay 已完成验收；
各次执行结果以本轮验证日志为准。
