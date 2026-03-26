# IsaacLab Navigation Extension - SRU 项目

[![Paper](https://img.shields.io/badge/IJRR-2025-blue)](https://journals.sagepub.com/home/ijr)
[![Website](https://img.shields.io/badge/Project-Website-green)](https://michaelfyang.github.io/sru-project-website/)

> **重要说明**：本仓库包含的是 SRU 项目的 **IsaacLab 任务扩展**，提供带有动态障碍物配置和地形变化的多样化导航环境。本仓库 **不** 包含 `rsl_rl` 学习模块（网络结构、PPO/MDPO 训练算法）。完整导航系统请参考[项目网站](https://michaelfyang.github.io/sru-project-website/)。

## 概述

这是一个面向 Isaac Lab v2.1.1（Isaac Sim 4.5）的独立、自包含视觉导航任务扩展。本仓库提供：

- **环境**：带有动态障碍物配置和地形变化的 IsaacLab 多样化导航环境
- **任务定义**：面向强化学习视觉导航的分层控制架构接口
- **仿真**：具备真实深度传感器噪声的高保真物理仿真

**说明**：本仓库聚焦于仿真环境和任务定义。RL 训练基础设施（神经网络结构、PPO/MDPO 算法）由独立的 `rsl_rl` 学习模块提供。

该扩展实现了一个用于视觉导航的分层控制架构：
- **高层策略**：以 5Hz 输出 SE2 速度指令 `(vx, vy, omega)`
- **低层策略**：预训练运动控制策略，以 50Hz 将速度指令转换为关节动作

该扩展完全自包含，已内置所需的机器人模型、材质以及预训练运动控制策略。

### 包含内容

- ✅ 用于视觉导航环境的 IsaacLab 任务扩展
- ✅ 结合课程学习的迷宫地形生成
- ✅ 自包含资源：机器人模型（USD）、运动控制策略、深度编码器
- ✅ 多种机器人平台：B2W（双足轮式）与 AoW-D（带轮 Anymal）
- ✅ 观测定义：深度图、本体感知、目标指令
- ✅ 奖励函数：到达目标、动作平滑、运动惩罚
- ✅ 分层动作接口：将 SE2 速度指令传递给低层控制器
- ✅ 域随机化：相机位姿、动作缩放、低通滤波器、传感器延迟
- ✅ 兼容 RSL-RL 的训练脚本（支持 PPO/MDPO 算法）

### 不包含内容

- ❌ `rsl_rl` 学习模块（网络结构、PPO/MDPO 训练算法）
- ❌ 高层导航策略的神经网络结构
- ❌ On-policy 强化学习训练算法（PPO/MDPO 实现）

**说明**：如需训练导航策略，必须额外安装 `rsl_rl` 包。见下方安装说明。

### 相关项目

- [sru-pytorch-spatial-learning](https://github.com/michaelfyang/sru-pytorch-spatial-learning) - SRU 核心架构
- [SRU Project Website](https://michaelfyang.github.io/sru-project-website/) - 完整导航系统

## 特性

- 使用深度相机进行**视觉导航**，并带有真实噪声仿真
- 支持课程学习的**迷宫地形生成**
- **自包含资源**：已包含所有机器人模型和运动控制策略
- **多种机器人平台**：
  - **B2W**：双足轮式机器人（配备 ZedX 相机）
  - **AoW-D**：带轮 Anymal（配备 ZedX 相机）
- 带特权 critic 观测的**非对称 actor-critic**
- 用于地形难度递进的**课程学习**
- **多种算法**：通过 RSL-RL 支持 MDPO 和 PPO
- **域随机化**：相机位姿、动作缩放、低通滤波器、传感器延迟

## 安装

### 前置条件

- 已安装并配置 Isaac Lab v2.1.1
- Isaac Sim 4.5.0
- Python 3.10
- PyTorch >= 2.5.1

### 第一步：克隆或放置扩展

该扩展应放在 IsaacLab 安装目录的 `source/` 目录下：

```bash
# 进入你的 IsaacLab 安装目录
cd /path/to/IsaacLab

# 如果你是单独克隆本仓库，请将其放到 source/ 下
# 目录结构应类似：
# IsaacLab/
# ├── source/
# │   ├── isaaclab/
# │   ├── isaaclab_assets/
# │   └── isaaclab_nav_task/  <- 本扩展
```

### 第二步：安装扩展

从 IsaacLab 根目录以开发模式安装该扩展：

```bash
# 在 IsaacLab 根目录下
./isaaclab.sh -p -m pip install -e source/isaaclab_nav_task

# 或者进入扩展目录
cd source/isaaclab_nav_task
../../isaaclab.sh -p -m pip install -e .
```

### 第三步：安装 RSL-RL（训练必需）

该扩展训练时依赖 `rsl_rl` 包。请安装包含 MDPO/PPO 算法的自定义版本：

```bash
# 克隆并安装自定义 rsl_rl（如果尚未安装）
cd /path/to/your/workspace
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
pip install -e .
```

### 验证安装

测试扩展是否已正确安装：

```bash
# 在 IsaacLab 根目录下
./isaaclab.sh -p -m pip show isaaclab_nav_task

# 查看可用任务
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py --help
```

你应该能看到任务 ID 列表（例如 `Isaac-Nav-MDPO-B2W-v0`、`Isaac-Nav-PPO-AoW-D-v0` 等）。

## 可用任务

### B2W
| Task ID | 说明 |
|---------|------|
| `Isaac-Nav-MDPO-B2W-v0` | MDPO 训练 |
| `Isaac-Nav-PPO-B2W-v0` | PPO 训练 |
| `Isaac-Nav-MDPO-B2W-Play-v0` | MDPO 回放 |
| `Isaac-Nav-PPO-B2W-Play-v0` | PPO 回放 |
| `Isaac-Nav-MDPO-B2W-Dev-v0` | MDPO 开发测试 |
| `Isaac-Nav-PPO-B2W-Dev-v0` | PPO 开发测试 |

### AoW-D
| Task ID | 说明 |
|---------|------|
| `Isaac-Nav-MDPO-AoW-D-v0` | MDPO 训练 |
| `Isaac-Nav-PPO-AoW-D-v0` | PPO 训练 |
| `Isaac-Nav-MDPO-AoW-D-Play-v0` | MDPO 回放 |
| `Isaac-Nav-PPO-AoW-D-Play-v0` | PPO 回放 |
| `Isaac-Nav-MDPO-AoW-D-Dev-v0` | MDPO 开发测试 |
| `Isaac-Nav-PPO-AoW-D-Dev-v0` | PPO 开发测试 |

## 补充文档

- [DRONE_TRAINING_DEFAULTS.md](DRONE_TRAINING_DEFAULTS.md) - 无人机训练默认动力学、控制器使用、噪声与强化学习参数
- [TERRAIN_AND_GOALS.md](TERRAIN_AND_GOALS.md) - 地形生成、目标采样与坐标系相关说明

## 训练

### 使用独立训练脚本

```bash
# 使用 PPO 训练 B2W
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 --num_envs 4096 --headless

# 使用 PPO 训练 AoW-D
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-AoW-D-v0 --num_envs 4096 --headless

# 自定义 wandb run name
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-MDPO-B2W-v0 --num_envs 4096 --headless \
    --run_name "experiment_v1_with_curriculum"

# 同时指定多个自定义参数进行训练
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-v0 --num_envs 2048 --headless \
    --run_name "large_training_run" --seed 42 --max_iterations 20000
```

### 开发 / 测试（更小配置，使用 tensorboard）

`-Dev-v0` 变体默认使用 tensorboard 记录，并将训练轮数降低到 300（普通训练为 15000），便于快速测试：

```bash
# 使用较少环境数进行快速测试
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py \
    --task Isaac-Nav-PPO-B2W-Dev-v0 --num_envs 32 --headless
```

### 使用标准 RSL-RL 工作流

```bash
# 使用 RSL-RL 训练
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Nav-MDPO-B2W-v0 --num_envs 4096
```

## 回放已训练策略

```bash
# 使用独立脚本进行回放
./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-MDPO-B2W-Play-v0 --num_envs 16

# 使用指定 checkpoint 回放
./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-MDPO-B2W-Play-v0 \
    --checkpoint /path/to/model.pt
```

## 架构

```
isaaclab_nav_task/
├── config/
│   └── extension.toml            # 扩展元数据
├── docs/
│   └── README.md                 # 本文件
├── scripts/
│   ├── train.py                  # 训练脚本
│   └── play.py                   # 回放脚本
├── setup.py                      # 安装脚本
├── pyproject.toml                # 构建配置
└── isaaclab_nav_task/
    ├── __init__.py               # 扩展入口
    └── navigation/
        ├── __init__.py
        ├── navigation_env_cfg.py         # 基础环境配置
        ├── assets/                       # 机器人配置与数据
        │   ├── __init__.py
        │   ├── b2w.py                    # B2W 机器人配置
        │   ├── aow_d.py                  # AoW-D 机器人配置
        │   └── data/                     # 自包含资源目录
        │       ├── Robots/               # 机器人 USD 模型与材质
        │       │   └── AoW-D/            # AoW-D 机器人资源
        │       │       ├── aow_d.usd     # 机器人 USD 模型
        │       │       └── Props/        # 材质与纹理
        │       └── Policies/             # 预训练模型
        │           ├── depth_encoder/    # VAE 深度编码器
        │           │   └── vae_pretrain_new.pth  (ZedX)
        │           └── locomotion/       # 低层运动控制策略
        │               ├── aow_d/        # policy_blind_3_1.pt (1.7 MB)
        │               └── b2w/          # policy_b2w_new_2.pt (2.0 MB)
        ├── config/
        │   ├── rl_cfg.py                 # 基础 RL 配置
        │   ├── b2w/
        │   │   ├── __init__.py           # 任务注册
        │   │   ├── navigation_env_cfg.py
        │   │   └── agents/
        │   │       └── rsl_rl_cfg.py
        │   └── aow_d/
        │       ├── __init__.py
        │       ├── navigation_env_cfg.py
        │       └── agents/
        │           └── rsl_rl_cfg.py
        ├── mdp/
        │   ├── observations.py       # 观测函数（13 个函数）
        │   ├── rewards.py            # 奖励函数（5 个函数）
        │   ├── terminations.py       # 终止条件（4 个函数）
        │   ├── curriculums.py        # 课程项（1 个函数）
        │   ├── events.py             # 域随机化事件（5 个函数）
        │   ├── depth_utils/          # 深度处理工具
        │   │   ├── __init__.py
        │   │   ├── camera_config.py      # 相机配置（ZedX）
        │   │   └── depth_noise_encoder.py # 基于 VAE 的深度编码器
        │   └── navigation/
        │       ├── goal_commands.py
        │       ├── goal_commands_cfg.py
        │       └── actions/
        │           ├── __init__.py
        │           ├── navigation_se2_actions.py
        │           └── navigation_se2_actions_cfg.py
        └── terrains/                # 自定义地形生成器
            ├── __init__.py
            ├── hf_terrains_maze.py      # 迷宫地形生成
            ├── hf_terrains_maze_cfg.py  # 迷宫地形配置
            ├── maze_config.py           # 迷宫参数
            └── patches.py               # TerrainImporter 补丁
```

## 兼容性

- **Isaac Lab**: v2.1.1
- **Isaac Sim**: 4.5.0
- **Python**: 3.10
- **PyTorch**: >= 2.5.1

## 自包含资源

该扩展包含了所有必要资源，不依赖外部资源仓库：

### 机器人模型（`assets/data/Robots/`）
- **AoW-D**：完整 USD 模型，包含材质和纹理
  - 当基础 `isaaclab_assets` 中不可用 AoW-D 时使用
  - 包含所有所需 Props 和材质纹理（11 张 baked 纹理）

### 预训练策略（`assets/data/Policies/`）

**深度编码器**（`depth_encoder/`）：
- `vae_pretrain_new.pth`：用于 B2W 和 AoW-D 的 ZedX 相机编码器
- VAE 架构，采用 RegNet 主干 + Feature Pyramid Network

**运动控制策略**（`locomotion/`）：
- `aow_d/policy_blind_3_1.pt`（1.7 MB）：AoW-D 轮式运动控制
- `b2w/policy_b2w_new_2.pt`（2.0 MB）：B2W 双足轮式运动控制

所有运动控制策略都已预训练，并由分层动作控制器加载使用。

## 关键组件

### 导航环境（`navigation_env_cfg.py`）
- 定义包含地形、机器人和传感器的场景
- 配置 policy 和 critic 的观测组
- 设置到达目标和运动惩罚相关奖励项
- 配置地形难度课程

### MDP 组件（`mdp/`）

**已清理并优化**，移除了未使用函数以提升可维护性：

- **observations.py**（13 个函数）：深度图处理、本体感知、目标方向、延迟缓冲
- **rewards.py**（5 个函数）：到达目标、动作平滑、运动惩罚
- **terminations.py**（4 个函数）：超时、碰撞检测、角度限制、到达目标
- **curriculums.py**（1 个函数）：后退惩罚调度
- **events.py**（5 个函数）：相机随机化、动作缩放、延迟缓冲管理

### 导航动作（`mdp/navigation/actions/`）
- 采用 SE2 速度指令的分层动作空间
- 与预训练低层运动控制策略集成

### 地形生成与目标采样（`terrains/`）

该扩展基于 Isaac Lab 的地形生成系统实现了自定义迷宫地形生成器，可提供多样化导航环境，并支持安全的目标点与出生点采样。

**关键特性：**
- **四种地形类型**：迷宫、非迷宫/随机、楼梯、坑洞
- **课程学习**：180 个地形，按 6 个难度等级组织
- **安全位置采样**：目标点 padding 为 0.5m，出生点 padding 为 0.6m
- **网格优化**：大规模训练时顶点数量减少约 80-99%
- **显式布尔掩码**：预计算有效位置，加速采样

**地形配置：**
- 网格：6 行（难度）× 30 列（变体）= 180 个地形
- 尺寸：每个地形 30m × 30m，分辨率 0.1m（300×300 cells）
- 比例：30% 迷宫，20% 随机，30% 楼梯，20% 坑洞

如需查看无人机任务的默认训练参数，请参考 [DRONE_TRAINING_DEFAULTS.md](DRONE_TRAINING_DEFAULTS.md)。

如需查看地形生成、目标点/出生点采样、坐标系以及实现细节的完整说明，请参考 [TERRAIN_AND_GOALS.md](TERRAIN_AND_GOALS.md)。

### 深度处理（`mdp/depth_utils/`）
- **DepthNoise**：使用基于视差的滤波模拟真实双目相机深度噪声
- **DepthNoiseEncoder**：基于 VAE 的深度编码器，使用 RegNet 主干和 Feature Pyramid Network
- **相机配置**：为不同相机类型提供预定义配置

| Camera | Robots | Resolution | Depth Range | Encoder |
|--------|--------|------------|-------------|---------|
| ZedX | B2W, AoW-D | 64x40 | 0.25-10.0m | `vae_pretrain_new.pth` |

### 自定义机器人资源（`assets/`）

机器人配置模块定义了与机器人相关的参数：

**B2W**（`b2w.py`）：
- 执行器配置（位置/速度控制）
- 初始关节状态
- USD 资源路径（来自基础 `isaaclab_assets`）

**AoW-D**（`aow_d.py`）：
- 带轮四足机器人的执行器配置
- 初始关节状态
- USD 资源路径（来自本地 `assets/data/Robots/AoW-D/`）
- 当基础资源中不可用时使用本地机器人模型

两套配置都能与分层导航控制器和预训练运动控制策略无缝集成。

## Docker 与集群部署

### Docker 修改内容

Dockerfile 包含：
1. **自定义 RSL-RL**：以 editable 模式安装自定义 `rsl_rl` 包
2. **Git safe directories**：避免容器中的所有权报错

### 快速开始流程

```bash
# 1. 构建 Docker 镜像
./docker/container.sh start --suffix nav

# 2. 推送到集群（会自动转换为 Singularity）
./docker/cluster/cluster_interface.sh push base-nav

# 3. 提交训练任务
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-PPO-B2W-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--headless"

# 4. 查看任务状态
squeue -u $USER
```

### 配置

**步骤 1**：在 `docker/` 目录下创建 `.env.base-nav` 配置：
```bash
cp docker/.env.base docker/.env.base-nav
```

**步骤 2**：部署前配置 `docker/cluster/.env.cluster`：
- 设置 `CLUSTER_PYTHON_EXECUTABLE=source/isaaclab_nav_task/scripts/train.py`
- 添加集群凭据和路径配置

**步骤 3**：在 `docker/cluster/submit_job_slurm.sh` 中添加集群特定的 module load：
```bash
module load eth_proxy  # ETH 集群网络访问所需
```

详细说明请参考 [IsaacLab cluster guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html#cluster-guide)。

### 训练示例

```bash
# B2W 使用 MDPO 训练（10k iterations）
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-MDPO-B2W-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--headless"

# B2W 自定义 run name
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-MDPO-B2W-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--run_name experiment_v1_b2w" \
    "--headless"

# AoW-D 使用 MDPO 训练（10k iterations）
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-MDPO-AoW-D-v0" \
    "--num_envs 2048" \
    "--max_iterations 10000" \
    "--headless"

# 使用 PPO 进行快速 dev 测试（300 iter，tensorboard）
./docker/cluster/cluster_interface.sh job base-nav \
    "--task Isaac-Nav-PPO-B2W-Dev-v0" \
    "--num_envs 32" \
    "--headless"
```

### 故障排查

**Git 所有权报错**：重建 Docker 镜像（已包含修复），或在容器中执行：
```bash
git config --global --add safe.directory '*'
```

**内存问题**：降低 `--num_envs`，或增大 `#SBATCH --mem-per-cpu`

## 许可证

MIT License，详见 [LICENSE](../LICENSE) 文件。

Copyright (c) 2025 Fan Yang, Per Frivik, Robotic Systems Lab, ETH Zurich

## 引用

如果你的研究使用了本代码库，请引用：

```bibtex
@article{yang2025sru,
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  title = {Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  doi = {10.1177/02783649251401926},
  url = {https://doi.org/10.1177/02783649251401926}
}
```

## 联系方式

**作者**：
- Fan Yang (fanyang1@ethz.ch)
- Per Frivik (pfrivik@ethz.ch)

**机构**：ETH Zurich Robotic Systems Lab
