# 无人机训练默认参数

本文档总结当前仓库中无人机任务在训练时使用的默认动力学、控制、噪声与强化学习参数。

本文重点覆盖以下任务族：

- `Isaac-Nav-PPO-Drone-Static-*`
- `Isaac-Nav-PPO-Drone-Static-SwarmCompat-*`
- `Isaac-Nav-IPPO-Drone-Swarm-Static-*`

## 范围说明

当前无人机任务里存在两条不同的执行链路：

- 单机与 `SwarmCompat` 训练使用 `DroneAccelAction` 和底层 `Controller`。
- swarm 训练使用 `DroneSwarmNavigationEnv`，直接写 root pose 和 root velocity，不复用同一套控制器与动作执行噪声链路。

因此，单机与 swarm 的默认训练行为并不完全相同。

## 任务族概览

| 任务族 | 主配置文件 | 执行链路 |
|---|---|---|
| 单机无人机 | `isaaclab_nav_task/navigation/config/drone/navigation_env_cfg.py` | `DroneAccelAction` + `Controller` |
| 兼容 swarm 观测的单机无人机 | `isaaclab_nav_task/navigation/config/drone/navigation_env_cfg.py` | 与单机相同，但保留队友观测通道并填 0 |
| swarm 无人机 | `isaaclab_nav_task/navigation/config/drone_swarm/swarm_env_cfg.py` | `DroneSwarmNavigationEnv` 直接写 pose / velocity |

## 控制频率与仿真步长

| 项目 | 单机 | SwarmCompat | Swarm |
|---|---|---|---|
| 高层规划频率 | `20 Hz` | `20 Hz` | `20 Hz` |
| 物理步长 `sim.dt` | `0.005 s` | `0.005 s` | `0.005 s` |
| 环境步长 | `0.05 s` | `0.05 s` | `0.05 s` |
| `decimation` | `int((1 / 0.005) / 20) = 10` | 与单机相同 | `int((1.0 / 0.005) / 20) = 10` |
| 单回合时长 | `60 s` | `60 s` | `60 s` |

## 动作接口默认值

| 项目 | 单机 | SwarmCompat | Swarm |
|---|---|---|---|
| 动作维度 | `3` | `3` | `3` |
| 动作含义 | 当前机体坐标系下的 `[ax, ay, yaw_rate]` | 与单机相同 | 3 维动作映射到平面速度与偏航角速度 |
| 动作缩放 | `scale=[1.0, 1.0, 1.5]`，`offset=[0, 0, 0]` | 与单机相同 | `action_scale=(2.5, 2.5, 1.5)` |
| 动作分布处理 | `use_raw_actions=True`，`policy_distr_type="gaussian"`，随后做 `tanh` 和缩放 | 与单机相同 | `tanh(action) * action_scale` |
| 最大平面加速度 | `1.0` | `1.0` | 没有单独暴露对应字段 |
| 最大平面速度 | `1.0` | `1.0` | `2.5` |
| 目标飞行高度 | `1.2` | `1.2` | `1.2` |

## 默认是否使用控制器

| 项目 | 单机 | SwarmCompat | Swarm |
|---|---|---|---|
| 默认使用控制器 | `是` | `是` | `否` |
| 对应开关 | `use_controller=True` | 与单机相同 | 没有对应控制器开关 |
| 控制器更新频率 | `controller_decimation=2`，即 `0.01 s`，约 `100 Hz` | 与单机相同 | 不适用 |
| 力 / 力矩施加方式 | `set_external_force_and_torque()` | 与单机相同 | 不使用 |
| 直接写 root pose / velocity | 只在不使用控制器的理想分支里出现 | 与单机相同 | 每步都使用 |

## 底层控制器默认参数

以下参数只适用于单机与 `SwarmCompat` 链路。

| 参数 | 默认值 |
|---|---|
| `kPp` | `[0.0, 0.0, 10.0]` |
| `kPv` | `[0.0, 0.0, 10.0]` |
| `kPR` | `[13.0, 13.0, 13.0]` |
| `kPw` | `[0.017, 0.01, 0.02]` |
| `kIw` | `[0.0, 0.0, 0.0]` |
| `kDw` | `[0.0, 0.0, 0.0]` |
| 角速度积分限幅 | `[0.1, 0.1, 0.1]` |
| 最大倾角参数 | `controller_k_max_ang=30.0` |
| 最大反馈 bodyrate | `13.0` |
| 最大角加速度 | `200.0` |

## 动力学与重力默认值

| 项目 | 单机 | SwarmCompat | Swarm |
|---|---|---|---|
| 资产默认重力设置 | 无人机资产默认 `disable_gravity=True` | 与单机相同 | 与单机相同 |
| 训练时实际重力设置 | `disable_gravity = not use_controller`，默认训练会开启重力 | 与单机相同 | 在 `__post_init__` 中显式对 5 架机都设置为 `disable_gravity=True` |
| 线性阻尼 | `0.2` | `0.2` | 共享无人机资产中的 `0.2` |
| 角阻尼 | `0.4` | `0.4` | 共享无人机资产中的 `0.4` |
| 陀螺力 | 开启 | 开启 | 开启 |

## 观测结构

| 项目 | 单机 | SwarmCompat | Swarm |
|---|---|---|---|
| Policy 核心本体观测 | 线速度、角速度、投影重力、上一动作、目标位置 | 与单机相同，再加队友特征 | 线速度、角速度、投影重力、上一动作、目标命令、队友特征 |
| 队友观测通道 | 无 | 有，固定 `24` 维，但默认填 0 | 有，使用真实队友观测 |
| Policy 特征维度 | `2576` | `2600` | `2600` |
| Critic 特征维度 | `5713` | `5737` | `5737` |

## 观测噪声与延迟

### 单机与 SwarmCompat

| 观测项 | 默认值 |
|---|---|
| 线速度噪声 | 均匀噪声 `[-0.2, 0.2]` |
| 角速度噪声 | 均匀噪声 `[-0.1, 0.1]` |
| 投影重力噪声 | 均匀噪声 `[-0.1, 0.1]` |
| 目标位置噪声 | `DeltaTransformationNoiseCfg(rotation=0.1, translation=0.5, noise_prob=0.1, remove_dist=False)` |
| 观测延迟管理器 | 开启 |
| 线速度最大延迟 | `2` step |
| 角速度最大延迟 | `2` step |
| 重力最大延迟 | `2` step |
| 目标位置最大延迟 | `2` step |
| 深度最大延迟 | `2` step |
| Policy 深度特征 | `depth_image_noisy_delayed` |
| Critic 深度特征 | `depth_image_prefect` |

### Swarm

| 观测项 | 默认值 |
|---|---|
| 显式本体加性噪声 | swarm 环境配置中未单独配置 |
| 观测延迟管理器 | `DroneSwarmNavigationEnv` 中未使用 |
| Policy 深度特征 | `depth_image_prefect` |
| Critic 深度特征 | `depth_image_prefect` |
| 队友观测缩放 curriculum | 默认开启 |

## 动作执行噪声与随机化

以下默认值只适用于单机与 `SwarmCompat` 执行链路。

| 项目 | 默认值 |
|---|---|
| 执行延迟 | 开启 |
| 延迟范围 | `(4, 6)` step |
| 动作 lag | 开启 |
| lag 时间常数 | `(0.04, 0.06) s` |
| XY 执行缩放随机化 | `(0.98, 1.02)` |
| Yaw 执行缩放随机化 | `(0.98, 1.02)` |
| XY 执行偏置随机化 | `(-0.02, 0.02)` |
| Yaw 执行偏置随机化 | `(-0.015, 0.015)` |
| XY 执行噪声 | `(-0.02, 0.02)` |
| Yaw 执行噪声 | `(-0.01, 0.01)` |

## 物理材质随机化

该 startup 事件默认配置在单机与 `SwarmCompat` 任务里。

| 项目 | 默认值 |
|---|---|
| 静摩擦系数范围 | `(0.8, 1.2)` |
| 动摩擦系数范围 | `(0.7, 1.0)` |
| 恢复系数范围 | `(0.0, 0.1)` |
| 分桶数量 | `64` |

swarm 环境在顶层配置中没有使用这套相同的事件式材质随机化块。

## reward 与终止默认值

### 单机与 SwarmCompat

| 项目 | 默认值 |
|---|---|
| 动作变化惩罚 | `-0.05` |
| guidance 进度奖励 | `+0.7` |
| 反向前进惩罚 | `-0.3` |
| guidance 横向误差惩罚 | `-0.25` |
| 回合终止惩罚 | `-50.0` |
| 软目标奖励 | `+0.25` |
| 紧目标奖励 | `+1.5` |
| body contact 终止阈值 | `0.01` |
| 超时距离阈值 | `0.5` |
| 地形跌落阈值 | `0.5` |

### Swarm

| 项目 | 默认值 |
|---|---|
| 进度奖励权重 | `0.7` |
| 反向惩罚权重 | `0.3` |
| 横向误差权重 | `0.25` |
| 软目标权重 | `0.25` |
| 紧目标权重 | `1.5` |
| 集群进度权重 | `0.15` |
| 集群到达 bonus 权重 | `1.25` |
| 进入目标区域奖励权重 | `2.0` |
| 成功奖励权重 | `4.0` |
| 过近惩罚权重 | `0.8` |
| 过远惩罚权重 | `0.35` |
| pairwise 过远惩罚权重 | `0.2` |
| 碰撞惩罚权重 | `15.0` |
| 接触惩罚权重 | `6.0` |
| 终止惩罚权重 | `25.0` |
| 超速惩罚权重 | `0.15` |
| 动作变化惩罚权重 | `0.05` |
| 接触力阈值 | `0.02` |
| 接触终止力阈值 | `0.08` |
| 接触终止步数 | `3` |

## Swarm curriculum 默认值

以下设置只适用于 swarm 训练。

| 项目 | 默认值 |
|---|---|
| 智能体数量 | `5` |
| 队友观测半径 | `6.0` |
| 队友观测缩放最小 / 最大值 | `0.05 / 1.0` |
| 队友观测 warmup / ramp | `100 / 1000` |
| swarm penalty 缩放最小 / 最大值 | `0.25 / 1.0` |
| swarm penalty warmup / ramp | `150 / 1200` |
| 集群碰撞终止 | 开启 |
| 集群碰撞终止模式 | `"curriculum"` |
| 集群碰撞终止 warmup | `1200` |

`SOLO` 版 swarm 任务会额外设置 `disable_teammate_observations=True` 和 `solo_pretraining=True`。

## RL Runner 默认值

| 项目 | 单机 / SwarmCompat | Swarm |
|---|---|---|
| Runner 类型 | PPO | 共享策略配置上的 IPPO |
| `num_steps_per_env` | `16` | `16` |
| 训练 `max_iterations` | `15000` | `15000` |
| Dev `max_iterations` | `300` | `300` |
| `save_interval` | `500` | `500` |
| `seed` | `60` | `60` |
| `reward_shifting_value` | `0.05` | `0.05` |
| `init_noise_std` | `1.0` | `1.0` |
| Actor hidden dims | `[512, 256, 128]` | `[512, 256, 128]` |
| Critic hidden dims | `[512, 256, 128]` | `[512, 256, 128]` |
| RNN 类型 | `lstm_sru` | `lstm_sru` |
| RNN hidden size | `512` | `512` |
| Dropout | `0.2` | `0.2` |
| 学习率 | `1e-3` | `1e-3` |
| `gamma` | `0.995` | `0.995` |
| `lam` | `0.95` | `0.95` |
| 熵系数 | `0.00375` | `0.00375` |

## 实际使用时的注意点

1. `SwarmCompat` 只是在观测维度上兼容 swarm 风格的队友通道，但动力学与控制仍然走单机链路。
2. 单机训练默认使用的观测噪声和动作执行噪声明显重于 swarm。
3. swarm 当前使用的是更理想化的动作执行方式，因为它直接写 root pose / velocity，而不是使用底层控制器。
4. 比较单机与 swarm 的 reward 曲线时要特别小心，因为两边的动作执行链路与噪声模型并不一致。

## 主要来源文件

- `isaaclab_nav_task/navigation/config/drone/navigation_env_cfg.py`
- `isaaclab_nav_task/navigation/config/drone_swarm/swarm_env_cfg.py`
- `isaaclab_nav_task/navigation/mdp/navigation/actions/drone_accel_actions_cfg.py`
- `isaaclab_nav_task/navigation/mdp/navigation/actions/drone_accel_actions.py`
- `isaaclab_nav_task/navigation/utils/controller.py`
- `isaaclab_nav_task/navigation/swarm_navigation_env.py`
- `isaaclab_nav_task/navigation/mdp/delay_manager.py`
- `isaaclab_nav_task/navigation/assets/drone.py`
- `isaaclab_nav_task/navigation/config/drone/agents/rsl_rl_cfg.py`
- `isaaclab_nav_task/navigation/config/drone_swarm/agents/rsl_rl_cfg.py`
