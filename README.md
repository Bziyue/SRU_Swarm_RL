# SRU Navigation Workspace

This parent repository is wired for SRU navigation training with:

- `sru-navigation-sim`: Isaac Lab task extension
- `sru-navigation-learning`: SRU-enhanced RL framework
- `sru-pytorch-spatial-learning`: standalone SRU memory experiments

## Expected Local Setup

The helper scripts are designed to work directly from this repository without hardcoded home-directory paths:

- they prefer the currently activated conda environment via `CONDA_PREFIX`
- if no env is active, they try to resolve `ENV_NAME` via `conda info --base`
- they try to locate Isaac Lab next to this repo, for example `../IsaacLab-*`
- they try to locate Isaac Sim in common sibling locations, for example `../isaacsim` or `../_isaac_sim`

If your layout differs, set environment variables explicitly:

```bash
CONDA_ENV_DIR=/path/to/conda/env ISAACLAB_DIR=/path/to/IsaacLab ISAACSIM_DIR=/path/to/isaacsim ./scripts/setup_nav_training.sh
```

## Direct Training From This Workspace

The helper scripts prepend this workspace to `PYTHONPATH`, so you can train from here without first modifying the Conda environment.

Recommended usage:

```bash
conda activate env_isaacsim
./scripts/check_nav_env.sh
./scripts/train_nav.sh
```

## Optional Persistent Setup

```bash
conda activate env_isaacsim
./scripts/setup_nav_training.sh
```

The setup script is only needed if you want `env_isaacsim` itself to permanently use these packages. It will:

- uninstall the preinstalled `rsl-rl-lib` package from the target env
- remove any stale bundled `rsl_rl` package under Isaac Sim if present
- install editable `sru-navigation-learning`
- install editable `sru-navigation-sim`

## Training Defaults

`./scripts/train_nav.sh` defaults to:

- task: `Isaac-Nav-PPO-B2W-Dev-v0`
- env count: `32`
- headless mode: enabled

Common overrides:

```bash
TASK=Isaac-Nav-MDPO-B2W-v0 NUM_ENVS=2048 ./scripts/train_nav.sh
RUN_NAME=debug_b2w MAX_ITERATIONS=300 ./scripts/train_nav.sh
./scripts/train_nav.sh Isaac-Nav-PPO-AoW-D-Dev-v0 --seed 42
```

If you need to move goal/spawn position-table precomputation off GPU on a smaller machine, set it in the task config:

```bash
self.commands.robot_goal.position_table_device = "cpu"
```

## Topology Guidance Rewards

The trajectory-guided drone task (`Isaac-Nav-PPO-Drone-Static-v0` and its `Dev`/`Play` variants) adds a topology guidance term on top of the original point-goal rewards.

### How the guidance is built

1. Region-to-region trajectories are generated offline in `Indoor-topology-generation` and exported to:
   - `sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Environments/StaticScan/all_region_pair_trajectories.json`
2. At environment initialization, `StaticRegionGoalCommand` loads the whole file once and precomputes all guidance centerlines.
3. Each quintic trajectory is:
   - densely evaluated with `guidance_trajectory_eval_dt = 0.05`
   - then resampled by arc length with `guidance_arc_length_spacing = 0.2`
4. The result is a smooth centerline used as a topology guide. Training does not recompute trajectories online; it only indexes the precomputed centerline that matches the sampled region pair.

In the current asset set, initialization precomputes:

- `2256` directed guidance trajectories
- `1128` undirected region pairs

### How an episode gets its guidance

For each reset:

1. A directed region pair is sampled.
2. A safe spawn point is sampled inside the source region.
3. A safe goal point is sampled inside the target region.
4. The matching precomputed guidance centerline is attached to that environment.

Safe points are sampled on the fixed-height plane:

- `flight_height = 1.2`
- safety clearance to mesh: `point_clearance = 0.15`
- safe-point grid spacing: `safe_point_grid_spacing = 0.25`

### Guidance reward terms

The drone task currently uses these topology guidance rewards:

- `guidance_progress`
  - weight: `1.0`
  - parameter: `clamp_delta = 0.5`
  - implementation: reward the change in projected arc-length progress along the current guidance centerline
  - formula:

```text
progress_delta = clamp(current_progress - previous_progress, -0.5, 0.5)
guidance_progress_reward = progress_delta / 0.5
```

- `guidance_lateral_error`
  - weight: `-0.15`
  - parameter: `sigma = 0.75`
  - implementation: penalize lateral deviation from the current guidance centerline
  - formula:

```text
guidance_lateral_error_penalty = tanh(lateral_error / 0.75)
```

Because the reward weight is negative, larger lateral deviation produces a larger penalty in the final reward.

### Full reward mix for the drone task

Topology guidance is combined with the existing navigation rewards:

- `action_rate_l1`: weight `-0.05`
- `guidance_progress`: weight `1.0`
- `guidance_lateral_error`: weight `-0.15`
- `episode_termination`: weight `-50.0`
- `reach_goal_xy_soft`: weight `0.25`
- `reach_goal_xy_tight`: weight `1.5`

So the current task is not pure trajectory tracking. It is still a point-goal navigation task, but now it also gets dense shaping from a precomputed topology-consistent smooth path between the sampled source and target regions.
