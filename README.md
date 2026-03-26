# SRU Navigation 工作区

这个父仓库已经按 SRU 导航训练工作流组织好，包含：

- `sru-navigation-sim`：Isaac Lab 任务扩展
- `sru-navigation-learning`：SRU 增强的强化学习框架
- `sru-pytorch-spatial-learning`：独立的 SRU 记忆实验

## 预期本地目录结构

这些辅助脚本被设计为可直接在当前仓库中运行，不依赖写死的 home 目录路径：

- 优先使用当前已激活的 conda 环境，通过 `CONDA_PREFIX` 获取
- 如果当前没有激活环境，则尝试通过 `conda info --base` 结合 `ENV_NAME` 解析
- 会尝试在当前仓库同级目录寻找 Isaac Lab，例如 `../IsaacLab-*`
- 会尝试在常见同级目录寻找 Isaac Sim，例如 `../isaacsim` 或 `../_isaac_sim`

如果你的目录结构不同，请显式设置环境变量：

```bash
CONDA_ENV_DIR=/path/to/conda/env ISAACLAB_DIR=/path/to/IsaacLab ISAACSIM_DIR=/path/to/isaacsim ./scripts/setup_nav_training.sh
```

## 直接在本工作区启动训练

辅助脚本会把当前工作区加入 `PYTHONPATH`，因此你可以直接从这里启动训练，而不必先修改 Conda 环境本身。

推荐使用方式：

```bash
conda activate env_isaacsim
./scripts/check_nav_env.sh
./scripts/train_nav.sh
```

## 可选的持久化安装

```bash
conda activate env_isaacsim
./scripts/setup_nav_training.sh
```

只有当你希望 `env_isaacsim` 这个环境长期、默认使用这些包时，才需要执行安装脚本。它会：

- 从目标环境中卸载预装的 `rsl-rl-lib`
- 如果存在，删除 Isaac Sim 下陈旧的内置 `rsl_rl` 包
- 以 editable 模式安装 `sru-navigation-learning`
- 以 editable 模式安装 `sru-navigation-sim`

## 训练默认值

`./scripts/train_nav.sh` 的默认设置为：

- task：`Isaac-Nav-PPO-B2W-Dev-v0`
- 环境数：`32`
- headless 模式：开启

常见覆盖方式：

```bash
TASK=Isaac-Nav-MDPO-B2W-v0 NUM_ENVS=2048 ./scripts/train_nav.sh
RUN_NAME=debug_b2w MAX_ITERATIONS=300 ./scripts/train_nav.sh
./scripts/train_nav.sh Isaac-Nav-PPO-AoW-D-Dev-v0 --seed 42
```

如果你的机器显存较小，需要把目标点 / 出生点位置表预计算从 GPU 挪到 CPU，可以在任务配置里设置：

```bash
self.commands.robot_goal.position_table_device = "cpu"
```

## 拓扑引导奖励

轨迹引导版无人机任务（`Isaac-Nav-PPO-Drone-Static-v0` 及其 `Dev` / `Play` 变体）会在原始 point-goal 奖励之外，额外加入一个拓扑引导项。

### 引导轨迹如何构建

1. 区域到区域的轨迹在 `Indoor-topology-generation` 中离线生成，并导出到：
   - `sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Environments/StaticScan/all_region_pair_trajectories.json`
2. 环境初始化时，`StaticRegionGoalCommand` 会一次性加载整个文件，并预计算所有引导中心线。
3. 每条五次轨迹会先：
   - 使用 `guidance_trajectory_eval_dt = 0.05` 做密集采样
   - 再按弧长使用 `guidance_arc_length_spacing = 0.2` 重采样
4. 最终得到一条平滑中心线，作为拓扑引导。训练时不会在线重新生成轨迹，只会索引与当前采样区域对相匹配的预计算中心线。

在当前资源配置下，初始化阶段会预计算：

- `2256` 条有向引导轨迹
- `1128` 个无向区域对

### 一个 episode 如何获得引导轨迹

每次 reset 时：

1. 先采样一个有向区域对。
2. 在起始区域内采样一个安全出生点。
3. 在目标区域内采样一个安全目标点。
4. 将匹配的预计算引导中心线绑定到该环境实例。

安全点采样都发生在固定高度平面上：

- `flight_height = 1.2`
- 与网格的安全间距：`point_clearance = 0.15`
- 安全点网格间距：`safe_point_grid_spacing = 0.25`

### 引导奖励项

当前无人机任务使用以下拓扑引导奖励：

- `guidance_progress`
  - 权重：`1.0`
  - 参数：`clamp_delta = 0.5`
  - 实现方式：奖励沿当前引导中心线投影弧长进度的增量
  - 公式：

```text
progress_delta = clamp(current_progress - previous_progress, -0.5, 0.5)
guidance_progress_reward = progress_delta / 0.5
```

- `guidance_lateral_error`
  - 权重：`-0.15`
  - 参数：`sigma = 0.75`
  - 实现方式：惩罚相对于当前引导中心线的横向偏离
  - 公式：

```text
guidance_lateral_error_penalty = tanh(lateral_error / 0.75)
```

由于该奖励项的权重为负，因此横向偏离越大，最终总奖励中的惩罚也越大。

### 无人机任务的完整奖励组合

拓扑引导与现有导航奖励共同组成最终奖励：

- `action_rate_l1`：权重 `-0.05`
- `guidance_progress`：权重 `1.0`
- `guidance_lateral_error`：权重 `-0.15`
- `episode_termination`：权重 `-50.0`
- `reach_goal_xy_soft`：权重 `0.25`
- `reach_goal_xy_tight`：权重 `1.5`

因此，当前任务并不是纯粹的轨迹跟踪任务。它本质上仍然是 point-goal 导航任务，只是在采样到的起点区域和目标区域之间，额外引入了一条预计算且拓扑一致的平滑路径，作为密集 shaping 信号。
