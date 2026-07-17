# Remote L4 NPU Smoke

这个目录用于临时验证：

```text
L4 parent -> remote L3 session on machineA -> machineA NPU 0 + NPU 1
          -> remote L3 session on machineB -> machineB NPU 0 + NPU 1
```

当前 smoke 复用 `tests/st/a2a3/tensormap_and_ringbuffer/test_l3_group.py`
的 vector group 计算：每台远端机器各自启动一个 L3 worker，并在本机两个 NPU 上
通过 `submit_next_level_group()` 跑同一个 AIV vector kernel group。parent 会把
输入 copy 到两台远端、触发 L4 remote dispatch、再 copy 回输出做 golden 校验。

它不验证跨机 NPU buffer import/export；那部分仍依赖后续 HCOMM 硬件数据面。

## 远端机器

每台机器 checkout 对应分支，安装后启动 daemon：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH="$ASCEND_HOME_PATH/bin:$PATH"

bash tools/remote_l4_npu/start_machine_daemon.sh
```

默认端口在 `tools/remote_l4_npu/machine.env`。daemon 本身不走 `task-submit`；
parent 会通过 remote session manifest 指定这台机器使用哪些 NPU。

```bash
SIMPLER_REMOTE_L4_NPU_PORT=19073 bash tools/remote_l4_npu/start_machine_daemon.sh
```

## Parent 机器

两台远端 daemon 都启动后，在 parent checkout 运行：

```bash
bash tools/remote_l4_npu/run_parent_smoke.sh
```

脚本默认使用：

```bash
--machine-a 120.9.10.37:19073
--machine-b 120.9.10.35:19073
--machine-a-devices 0,1
--machine-b-devices 0,1
```

如果设备号不同，用环境变量覆盖：

```bash
SIMPLER_REMOTE_L4_NPU_MACHINE_A_DEVICES=2,3 \
SIMPLER_REMOTE_L4_NPU_MACHINE_B_DEVICES=2,3 \
bash tools/remote_l4_npu/run_parent_smoke.sh
```

成功时 parent 会打印两台 remote worker 都完成 L3 group NPU 计算和 golden 校验。
