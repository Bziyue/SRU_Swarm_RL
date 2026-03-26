# voxel_guidance_ros2

Standalone ROS 2 package for:

- loading `test/flysite.ply`
- voxelizing at `0.2m`
- inflating occupied voxels by `0.2m`
- publishing inflated voxels as `PointCloud2` to RViz
- publishing region centers and labels from `DR_Surface_BBox_Data.txt`
- running one 26-neighborhood Dijkstra per source region on the full voxel-map AABB
- generating all undirected region-pair paths
- constraining region-anchor search to a shrunken region AABB via `region_shrink_x / region_shrink_y`
- using a unified world-coordinate z band `min_search_world_z .. max_search_world_z`
- applying the same world-z band to anchor selection, Dijkstra expansion, and path shortening
- backtracking, shortening, resampling, and saving consolidated path text data
- running GCOPTER-based quintic spline optimization after path export
- using the pre-resample shortened path as the front-end guide route for trajectory optimization
- saving optimized polynomial breakpoints, coefficients, and safe-flight-corridor geometry to JSON

## Build

Put this package in a ROS 2 workspace `src/` directory, then:

```bash
colcon build --packages-select voxel_guidance_ros2
source install/setup.bash
```

## Run

```bash
ros2 run voxel_guidance_ros2 voxel_guidance_node
```

Useful params:

```bash
ros2 run voxel_guidance_ros2 voxel_guidance_node --ros-args \
  -p publish_stride:=8 \
  -p max_expansions:=4000000 \
  -p output_dir:=/tmp/region_paths_txt \
  -p output_file:=all_region_pair_paths.txt \
  -p trajectory_output_file:=all_region_pair_trajectories.json \
  -p enable_trajectory_optimization:=true \
  -p min_search_world_z:=1.5 \
  -p max_search_world_z:=2.0 \
  -p region_shrink_x:=1.0 \
  -p region_shrink_y:=1.0
```

Key trajectory params:

```bash
-p corridor_progress:=7.0 \
-p corridor_range:=3.0 \
-p trajectory_time_weight:=20.0 \
-p trajectory_integral_intervals:=16 \
-p trajectory_rel_cost_tol:=1e-5 \
-p trajectory_max_vel_mag:=1.0 \
-p trajectory_max_bdr_mag:=2.1
```

Outputs:

- `all_region_pair_paths.txt`: region-pair guidance paths
- `all_region_pair_trajectories.json`: optimized quintic spline breakpoints, polynomial coefficients, and safe corridor polyhedra

## Topics

- `/voxel_cloud`
- `/region_markers`
