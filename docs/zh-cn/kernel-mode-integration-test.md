# Kernel 模式集成测试说明

本文说明 kernel 模式集成线上的端到端测试是怎么做的：调用方侧如何准备设备和
stream，simpler 的四个入口各做了什么，一次 launch 在三条 stream 上怎样排布，
设备侧如何执行，以及图模式目前验证到哪一步。

整合了哪些 PR、每个 PR 冻结在哪个 head、跑过哪些测试，见
[集成验证记录](../kernel-integration-validation.md)。逐条冲突裁决见仓库根目录的
`INTEGRATION-LOG.md`。

## 1. 被测的是什么

kernel 模式下，simpler 是一个被调用的库：它借用调用方已经设好的设备和调用方
自己的 stream，提交一个有界的异步算子，不自己初始化 ACL，不 reset 设备，launch
路径上不做任何同步。这正是它区别于 program 模式的地方。

端到端数值用例是
`tests/ut/py/test_kernel_mode_c_api.py` 里的
`test_kernel_eager_launch_executes_fresh_tensor_and_scalar_snapshots`。它用 ctypes
直接调用 host runtime 动态库，自己扮演调用方，不经过 PyTorch，也不经过 simpler
的 Python Worker。

被执行的算子是一个 AIV 向量加标量，`y[i] = x[i] + scalar`：

| 组成 | 源文件 | 编译产物 |
| ---- | ------ | -------- |
| 编排 | `tests/ut/py/kernel_eager_orchestration.cpp` | AICPU 上运行的 .so，提交一个 AIV 任务 |
| 设备算子 | `examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp` | AIV 核上的代码段 |

两者在测试开头现场编译，打包成一个 `ChipCallable`，签名依次是输入、输出、标量。
单个 tensor 是 16,384 个 FLOAT32，64 KB。

## 2. 怎么运行

该用例带 `requires_hardware`、`platforms(["a2a3"])`、`runtime("tensormap_and_ringbuffer")`
和 `device_count(1)` 标记，只能在 a2a3 真机上运行。仿真平台的 kernel init 返回
`PTO_RUNTIME_ERR_UNSUPPORTED`。

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3
task-submit --device auto --device-num 1 \
    --run "python -m pytest tests/ut/py/test_kernel_mode_c_api.py --platform a2a3 --device \$TASK_DEVICE"
```

pytest 用例本身只是一个壳。它读取分配到的卡号，再启动一个子进程运行同一个文件，
参数为 `a2a3 tensormap_and_ringbuffer <卡号> eager_values`。放进子进程，是为了让
`aclInit` / `aclFinalize` 这类进程级状态不影响 pytest 进程。

## 3. 调用方侧：设备、stream 与显存

以下调用全部由测试完成，simpler 不代劳。ACL 符号从已加载的 `libhost_runtime.so`
句柄上解析，实际命中它依赖的 CANN 库。

```text
aclInit(NULL)
aclrtSetDevice(卡号)                 # 把这张卡设为当前线程的 current device
aclrtCreateStream(&caller_stream)    # 调用方自己的 stream
create_device_context()              # 只在 host 上构造对象，不接触设备
```

每一轮，测试用 `aclrtMalloc` 分配输入、输出两块显存，用 `aclrtMemcpy` 把输入拷到
设备，并把输出预填为 `-999`。

收尾顺序同样由测试控制：`finalize_device`、`destroy_device_context`、`aclrtFree`、
`aclrtDestroyStream`、`aclrtResetDevice`、`aclFinalize`。simpler 的 finalize 只释放
自己申请的资源。

同文件的生命周期用例用 `LD_PRELOAD` 拦截 `aclInit`、`aclrtSetDevice`、
`aclrtResetDevice`、`aclFinalize`、`rtDeviceReset` 等函数并计数，断言 kernel 模式下
simpler 没有调用过它们。

## 4. simpler 的四个入口

| 入口 | 做什么 |
| ---- | ------ |
| `simpler_kernel_mode_init` | 用 `aclrtGetDevice` 确认当前卡与传入卡号一致，只核对不设置；用 `rtStreamCreate` 创建两条私有 stream（AICPU、AICore）和 5 个 event；加载 AICPU 执行体。容量配置此时固定 |
| `simpler_kernel_mode_prepare_callable` | 上传 callable 镜像，把编排 .so 注册到设备；首次调用时提交 pooled arena，上传常驻 Runtime 与 KernelArgs |
| `simpler_kernel_mode_launch` | 校验上下文与 callable 驻留，编码本次参数包，在三条 stream 上排布一串异步操作后返回 |
| `finalize_device` | 释放上下文拥有的资源；测试断言 committed memory 归零 |

init 期间的执行体加载和 prepare 期间的 callable 注册，会同步上下文自己的 AICPU
stream。launch 路径不同步任何 stream。

prepare 的注册同步错误直接由该调用返回。eager 测试保留了额外 caller stream 同步，
但后面的 cold capture 测试不依赖这一步，也不要求先执行 eager warmup。

## 5. 一次 launch 的时序

launch 在 `src/common/platform/onboard/host/kernel_launch_owner.cpp` 中实现，把参数包交给
binder，在三条 stream 上排布如下操作：

```text
caller stream              AICore 私有 stream             AICPU 私有 stream
清零握手区与 gate
record(Start)
                           wait(Start)
                           rtKernelLaunchWithHandleV2
                           record(AicoreDone)
                                                          wait(Start)
                                                          rtsLaunchCpuKernel(参数包)
                                                          record(AicpuDone)
wait(AicpuDone)
wait(AicoreDone)
record(SerialTail)
```

launch 只负责入队，全部入队成功即返回 0，此时设备上可能尚未开始执行。caller
stream 上排有等待两个 done 事件的操作，所以调用方同步 caller stream 时，会一直
等到 AICPU 与 AICore 两侧都完成。内部不同步、错误由调用方同步暴露，就是这样实现的。

AICore 先于 AICPU 提交：AICPU 上的调度器会自旋等待 AICore 的握手，AICore 的任务
必须先排在它的 stream 上。

## 6. 设备侧执行

AICPU 入口 `simpler_aicpu_kernel_exec` 收到参数包后：

1. 校验包头，读取 callable 的驻留描述符并比对 generation，不匹配即拒绝。
2. 进入 TMR 执行路径，从参数包中的 binding 地址读出常驻 KernelArgs 与 Runtime。
3. 设置平台寄存器，由 leader 线程接纳本次调用，多个 AICPU 线程在 barrier 汇合。
4. 运行编排函数，编排提交的 AIV 任务经握手区派发给 AICore。
5. AICore 执行加标量，结果写入测试分配的输出显存。

## 7. 断言了什么

| 顺序 | 断言 |
| ---- | ---- |
| init 后 | `simpler_kernel_mode_supported(ctx) == 1` |
| prepare 并同步后 | context 的 committed memory 大于 0 |
| 每轮 launch | 返回 0；返回后立即清空主机侧参数对象 |
| 每轮同步后 | 输出与 `x + scalar` 逐元素精确相等，覆盖全部 16,384 个元素 |
| 每轮 | committed memory 与 prepare 后相同 |
| 第二轮后 | 第一轮输出仍然正确，未被覆盖 |
| finalize 后 | committed memory 为 0 |

两轮的输入与标量：

| 轮次 | 输入第 i 个元素 | scalar |
| ---- | --------------- | ------ |
| 1 | `i % 127` | 1.25 |
| 2 | `i % 127 + 257` | -3.5 |

launch 返回后立即清空主机侧参数，验证后续执行不借用调用者的参数对象。
这一步本身不能证明 CANN 不借用 Simpler 内部的编码缓冲；两图使用同一 callable、
不同地址/scalar，并在编码缓存被后一次 capture 改写后交替 replay，才覆盖该边界。

## 8. 图模式验证到哪一步

上面的数值用例是 eager 调用，不包含任何 `aclmdlRICapture` 调用。

独立探针 `tests/st/a2a3/kernel_capture/` 的验证范围是：

1. 在一条 warmup stream 上 eager 执行一遍。
2. 在 caller stream 上 `aclmdlRICaptureBegin`，执行同一串操作，`aclmdlRICaptureEnd` 得到图。
3. `aclmdlRIExecuteAsync` 回放 100 次，每次更换输入数据并校验结果，证明回放时内部 kernel 确实重新执行。
4. 用链接器 `--wrap` 拦截 `rtStreamAddToModel`、`rtStreamGetCaptureInfo`、`aclmdlRICaptureGetInfo`，断言调用次数为零。

该探针不经过公开的四个入口。它直接构造执行状态对象，自行加载 AICPU 执行体，
手写 record / wait 序列，使用测试专用的小 kernel，不经过 TMR runtime，也不经过
launch owner 与 binder。

真实 TMR 入口另由 `test_kernel_graph_capture_replays_public_launch` 验证，
复用上述 scalar-add callable，在 capture 内调用 `simpler_kernel_mode_launch`。
回放循环只调用 `aclmdlRIExecuteAsync`，不再次调用 Host launch。A2/A3 真机已通过：

| 场景 | Host launch 次数 | replay 验证 |
| ---- | ---------------- | ----------- |
| cold | capture 内 1 次；没有 eager warmup 或额外 prepare 后 caller 同步 | 100 次，输入地址不变、内容每次变化 |
| warm | eager 1 次、capture 内 1 次 | 100 次，逐元素精确比对 |
| A/B/A | 同 callable 分别 capture 两张图，共 2 次 | 两组地址和不同 scalar，按 A/B/A 顺序共 100 次；非活动输出不能被覆盖 |
| chain | 同一张图内 2 次 | `x → intermediate → output`，100 次；每轮毒化中间值和输出，同时检查两个节点的结果 |
| batch | 两张图各 2 次，共 4 次 | 交替提交 100 次 replay，循环中只有图执行 API，批末只同步一次；检查反馈数值与各图执行计数 |

原始集成快照的 cold 场景曾返回 `107024`：capture 内等待的 `PrepareTail`，
对应 record 却发生在 capture 外。eager warmup 会消费这条依赖，因此 warm 场景
无法暴露问题。prepare 在私有 AICPU stream 上完成同步注册后，production owner
不再传递该依赖；公开五参 ABI、launch 的异步 fork/join 顺序均不变。

除 batch 外，每轮 replay 后同步 caller 再读结果。batch 使用图 A 的 `y=x+1.25`、
图 B 的 `x=y+2.75` 形成反馈，各图还用当前支持原地读写的小算子累加独立计数。
从 `x=x0`、`y=-999` 和零计数出发，A/B 交替各执行 50 次后，
必须得到 `x=x0+200`、`y=x0+197.25`、两个计数均为 50。Host oracle 测试
逐位置验证单次漏执行、重复和相邻换序的可检出性；不声称覆盖任意组合故障。

测试专用 `kernel_capture_observer.cpp` 使用正式 CANN 类型，只读观察并原样转发
native AICore/AICPU launch，检查 KernelArgs 指针与 dispatch/K4 binding 相等且不变。
capture 外、设备已静止时，按正式 KernelArgs 类型 D2H 读取 `runtime_args`、`regs`、
`ffts_base_addr`，比较 replay 前后的静态地址。batch 提交循环内不做这类读取，
也不做 memcpy、查询或同步。replay 前后 native launch 计数不增长；这不是每次
replay 都回 Host 观察，更不是读取整个 Runtime 或 arena 的验证。

prepare 后、capture 后、串行每轮及 batch 完成后的 `committed_device_memory_ctx` 相等，
所有 graph 销毁之后才允许 finalize 和释放 tensor，正常 finalize 后 committed 为零。
这项计量不等于拦截过全部底层 alloc/free，也不包含 CANN 自己持有的 graph 内存。

## 9. 并入主线时的接口裁决

集成线并入当前主线（已含 #2064）时，按以下原则处理分歧：已合入的 K1 决定契约，
集成线保留主线骨架中没有的实现。

| 事项 | 结论 |
| ---- | ---- |
| 主机错误码编号 | 采用主线编号：`INVALID_ARGUMENT` 为 `BASE - 4`；集成线新增的 callable 与容量错误码顺延为 `BASE - 5` 至 `BASE - 9` |
| launch 的 callable id 越界 | 返回 `INVALID_ARGUMENT`，与 prepare 一致 |
| kernel 入口符号解析 | 每个 runtime 仍导出全部四个入口；ChipWorker 只在 `supported` 非零时解析 init、prepare、launch |
| kernel 模式容量规则 | 使用 K3 的共享 static arena bank，同时覆盖 onboard 与仿真 |
| 同步语义说明 | init 与 prepare 会同步上下文自己的 AICPU stream，launch 路径不同步 |

完整错误码表：

| 名称 | 值 |
| ---- | -- |
| `PTO_RUNTIME_ERR_INTERNAL` | -1000 |
| `PTO_RUNTIME_ERR_UNSUPPORTED` | -1001 |
| `PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE` | -1002 |
| `PTO_RUNTIME_ERR_INVALID_STATE` | -1003 |
| `PTO_RUNTIME_ERR_INVALID_ARGUMENT` | -1004 |
| `PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED` | -1005 |
| `PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED` | -1006 |
| `PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT` | -1007 |
| `PTO_RUNTIME_ERR_CALLABLE_STALE` | -1008 |
| `PTO_RUNTIME_ERR_CAPACITY_EXCEEDED` | -1009 |

## 10. 已知边界

- 公开 HBG kernel 执行尚未打通。H4 没有提交 PR，HBG 的 kernel 能力位为 0，init 返回 `UNSUPPORTED`。
- A5 只经过编译、单元测试与仿真，没有真机结果。
- 数值用例复用固定形状的 scalar-add callable，覆盖单节点、同图两节点及有序多图反馈，不代表任意算子或形状。
- 公开 capture/replay 只覆盖第 8 节的单 context、单 caller、固定形状场景；连续入队仍由同一 stream 保序，不代表允许并发共享 workspace。未验证任意跨流/跨图重叠、原生错误取消或 vLLM/PyPTO 混合执行。
- 同一 host runtime 动态库内限制同一设备只能有一个活动 kernel 上下文；跨动态库副本或跨进程需要调用方自行串行化。
