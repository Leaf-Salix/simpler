# Remote L4 NPU Smoke

这个目录用于临时验证：

```text
L4 parent -> remote L3 session on machineA -> machineA NPU
          -> remote L3 session on machineB -> machineB NPU
```

当前 smoke 只验证两台远端机器各自能收到 L4 派发并运行本机 NPU 上的空参
`ChipCallable`。它不验证跨机 NPU buffer import/export；那部分仍依赖后续 HCOMM
硬件数据面。

## 远端机器

每台机器 checkout 对应分支，安装后启动 daemon：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .

bash tools/remote_l4_npu/start_machine_daemon.sh
```

默认端口和设备在 `tools/remote_l4_npu/machine.env`。如果默认 `device=0` 不合适，
改环境变量即可：

```bash
SIMPLER_REMOTE_L4_NPU_DEVICE=3 bash tools/remote_l4_npu/start_machine_daemon.sh
```

## Parent 机器

两台远端 daemon 都启动后，在任一 checkout 运行：

```bash
export SIMPLER_REMOTE_L4_NPU_MACHINE_A=machine-a-host:19073
export SIMPLER_REMOTE_L4_NPU_MACHINE_B=machine-b-host:19073
export SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICE=0
export SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICE=0

bash tools/remote_l4_npu/run_parent_smoke.sh
```

成功时 parent 会打印两个 remote worker 都完成 NPU no-op dispatch。
