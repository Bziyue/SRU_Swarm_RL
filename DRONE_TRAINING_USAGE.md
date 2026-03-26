# 无人机训练使用说明

本文档面向当前工作区下的无人机任务训练使用场景。

默认约定：

- 已正确安装 Isaac Lab、Isaac Sim 与相关 Python 依赖
- 使用 `env_isaacsim` 作为训练环境
- 命令在仓库根目录执行
- 默认沿用仓库当前代码中的环境配置，包括控制器、动作延迟、观测延迟、噪声与其他训练随机化设置

## 进入环境

如果当前 shell 还没有加载 conda，可以先执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_isaacsim
```

如果你已经激活了 `env_isaacsim`，可以直接在仓库根目录执行后续命令。

## 单机预训练：SwarmCompat 版本

这个任务会保留队友观察输入通道，但实际输入置为 0，适合做与 swarm 网络结构兼容的单机预训练。

```bash
HEADLESS=1 NUM_ENVS=4096 RUN_NAME=single_pretrain_swarmcompat ./scripts/train_nav.sh Isaac-Nav-PPO-Drone-Static-SwarmCompat-v0
```

说明：

- `HEADLESS=1`：无界面训练
- `NUM_ENVS=4096`：并行环境数
- `RUN_NAME=single_pretrain_swarmcompat`：训练运行名称
- `Isaac-Nav-PPO-Drone-Static-SwarmCompat-v0`：单机预训练兼容版任务 ID

## 单机预训练：不带队友观察输入

如果你希望使用标准单机输入，不保留队友观察通道，可使用：

```bash
HEADLESS=1 NUM_ENVS=4096 RUN_NAME=single_pretrain_no_teammate ./scripts/train_nav.sh Isaac-Nav-PPO-Drone-Static-v0
```

## 快速开发测试

如果你想先做短程检查，而不是直接启动完整训练，推荐使用 `Dev` 版本：

```bash
HEADLESS=1 NUM_ENVS=64 MAX_ITERATIONS=300 RUN_NAME=dev_swarmcompat ./scripts/train_nav.sh Isaac-Nav-PPO-Drone-Static-SwarmCompat-Dev-v0
```

标准单机 `Dev` 版本：

```bash
HEADLESS=1 NUM_ENVS=64 MAX_ITERATIONS=300 RUN_NAME=dev_no_teammate ./scripts/train_nav.sh Isaac-Nav-PPO-Drone-Static-Dev-v0
```

## 常用可调参数

可以通过环境变量快速覆盖常见训练参数：

```bash
HEADLESS=1 \
NUM_ENVS=2048 \
MAX_ITERATIONS=15000 \
RUN_NAME=my_experiment \
SEED=42 \
./scripts/train_nav.sh Isaac-Nav-PPO-Drone-Static-SwarmCompat-v0
```

常见参数说明：

- `HEADLESS`：是否无界面运行，`1` 为开启，`0` 为关闭
- `NUM_ENVS`：并行环境数量
- `MAX_ITERATIONS`：最大训练轮数
- `RUN_NAME`：日志目录名与实验标识
- `SEED`：随机种子

## 训练日志

训练启动后，日志会写入 `logs/rsl_rl/` 下对应任务目录。通常可以重点查看：

- `console.log`
- TensorBoard 日志
- 保存的模型 checkpoint

## 备注

- `SwarmCompat` 单机任务默认保留 `teammate_features` 通道，但输入为 0。
- 标准单机任务不包含该队友观察输入。
- 本文档中的命令不会显式改动控制器、延迟或噪声参数，而是直接使用当前仓库代码中的默认配置。
