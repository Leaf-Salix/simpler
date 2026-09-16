# K7 caller 错误传播候选节点探针

验证 feat 的真实链式事件序列，在 JoinAicpu 后、SerialTail 前加入 caller 单线程检查节点，能否让调用方感知 hidden AICPU 的 native 0/2 结果。生产入口已接入对应检查节点；本探针仅验证传输机制，不替代生产错误路径验收。

- 调用正式 LoadAicpuOp / rtsLaunchCpuKernel 和生产 enqueue_kernel_launch_sequence；不安装或使用 Torch。
- prepare 分配固定 Device report 和锁页 Host 取证区；提交时不分配、不同步。测试末尾的有界同步只用于验收。
- hidden 主入口仍返回 native=2，不改成成功以强迫 checker 执行。
- Core 分支只有一项异步 memset，验证事件传播，不验证真实 AICore kernel 或 retirement。
- 每进程只做一轮 success/error，eager/replay 和 check/no-check 分别运行；终端错误后直接退出，不 reset、不 free、不复用上下文。
- success 必须有完成回执；error 既要 caller 返回预期 SDK 错误，也要原始执行及 checker 回执。D2H 不可读、超时或意外错误均为失败／未验证。
- 本探针不修改 borrowed stream 的 failure mode，不证明同图／跨图重叠准入。

构建时显式设置 checkout-local CCACHE_DIR、四路并行及 ASCEND_HOME_PATH。device 使用 CANN hcc 的 aarch64-target-linux-gnu-g++；host 使用服务器本机编译器。先做架构预检，然后单卡 task-submit，输入依次为 DEVICE、dispatcher SO、probe SO、eager|replay、check|no-check、success|error。
