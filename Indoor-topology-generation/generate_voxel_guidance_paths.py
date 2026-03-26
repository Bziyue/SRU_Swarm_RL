#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from path_utils import resolve_project_path


PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}


@dataclass(frozen=True)
class Region:
    region_id: int
    name: str
    corners: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    @property
    def bounds_min(self) -> np.ndarray:
        return self.corners.min(axis=0)

    @property
    def bounds_max(self) -> np.ndarray:
        return self.corners.max(axis=0)


def save_region_guidance_paths(
    output_path: str,
    region_paths: Dict[Tuple[int, int], np.ndarray],
    region_names: List[str] | None = None,
) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for (start_region_id, target_region_id), path_points in region_paths.items():
        arrays[f"path_{start_region_id}_{target_region_id}"] = np.asarray(path_points, dtype=np.float32)

    if region_names is not None:
        arrays["region_names"] = np.asarray(region_names, dtype=object)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **arrays)


def parse_region_boxes(bbox_path: Path) -> List[Region]:
    regions: List[Region] = []
    current_name: Optional[str] = None
    current_corners: List[np.ndarray] = []

    for raw_line in bbox_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Rectangle:"):
            if current_name is not None and current_corners:
                regions.append(
                    Region(
                        region_id=len(regions),
                        name=current_name,
                        corners=np.asarray(current_corners, dtype=np.float32),
                    )
                )
            current_name = line.split("Rectangle:", 1)[1].strip()
            current_corners = []
            continue

        if "Corner" in line and "X:" in line and "Y:" in line and "Z:" in line:
            tokens = line.replace(",", "").split()
            try:
                x = float(tokens[tokens.index("X:") + 1])
                y = float(tokens[tokens.index("Y:") + 1])
                z = float(tokens[tokens.index("Z:") + 1])
            except (ValueError, IndexError) as exc:
                raise RuntimeError(f"Failed to parse region corner line: {raw_line}") from exc
            current_corners.append(np.array([x, y, z], dtype=np.float32))

    if current_name is not None and current_corners:
        regions.append(
            Region(
                region_id=len(regions),
                name=current_name,
                corners=np.asarray(current_corners, dtype=np.float32),
            )
        )

    if not regions:
        raise RuntimeError(f"No regions parsed from {bbox_path}")
    return regions


def load_binary_ply_vertices(ply_path: Path) -> np.ndarray:
    with ply_path.open("rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {ply_path}")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if header_lines[0] != "ply":
            raise RuntimeError(f"Invalid PLY header: {ply_path}")

        format_line = next((line for line in header_lines if line.startswith("format ")), None)
        if format_line is None or "binary_little_endian" not in format_line:
            raise RuntimeError(f"Only binary little-endian PLY is supported: {ply_path}")

        vertex_count = 0
        vertex_properties: List[Tuple[str, str]] = []
        in_vertex_element = False
        for line in header_lines[1:]:
            if line.startswith("element "):
                parts = line.split()
                in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
                continue

            if in_vertex_element and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3:
                    raise RuntimeError(f"Unsupported PLY property line: {line}")
                prop_type, prop_name = parts[1], parts[2]
                if prop_type not in PLY_TYPE_TO_DTYPE:
                    raise RuntimeError(f"Unsupported PLY property type: {prop_type}")
                vertex_properties.append((prop_name, "<" + PLY_TYPE_TO_DTYPE[prop_type]))

        if vertex_count <= 0 or not vertex_properties:
            raise RuntimeError(f"Failed to parse vertex layout from {ply_path}")

        dtype = np.dtype(vertex_properties)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
        if not {"x", "y", "z"}.issubset(dtype.fields.keys()):
            raise RuntimeError(f"PLY does not contain x/y/z fields: {ply_path}")

        points = np.stack(
            [
                np.asarray(data["x"], dtype=np.float32),
                np.asarray(data["y"], dtype=np.float32),
                np.asarray(data["z"], dtype=np.float32),
            ],
            axis=1,
        )
    return points


def world_to_grid(points: np.ndarray, origin: np.ndarray, resolution: float) -> np.ndarray:
    return np.floor((points - origin[None, :]) / resolution).astype(np.int32)


def grid_to_world(indices: np.ndarray, origin: np.ndarray, resolution: float) -> np.ndarray:
    return origin[None, :] + (indices.astype(np.float32) + 0.5) * resolution


def build_occupancy_grid(
    points: np.ndarray,
    regions: List[Region],
    resolution: float,
    padding: float,
) -> Tuple[np.ndarray, np.ndarray]:
    region_points = np.stack([region.corners for region in regions], axis=0).reshape(-1, 3)
    global_min = np.minimum(points.min(axis=0), region_points.min(axis=0)) - padding
    global_max = np.maximum(points.max(axis=0), region_points.max(axis=0)) + padding

    shape = np.ceil((global_max - global_min) / resolution).astype(np.int32) + 1
    occupancy = np.zeros(shape.tolist(), dtype=bool)

    point_indices = world_to_grid(points, global_min, resolution)
    valid_mask = np.all(point_indices >= 0, axis=1) & np.all(point_indices < shape[None, :], axis=1)
    point_indices = np.unique(point_indices[valid_mask], axis=0)
    occupancy[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]] = True
    return occupancy, global_min


def inflate_occupancy(occupancy: np.ndarray, inflate_radius: float, resolution: float) -> np.ndarray:
    radius_voxels = int(math.ceil(inflate_radius / resolution))
    offsets = []
    for dx in range(-radius_voxels, radius_voxels + 1):
        for dy in range(-radius_voxels, radius_voxels + 1):
            for dz in range(-radius_voxels, radius_voxels + 1):
                if math.sqrt(dx * dx + dy * dy + dz * dz) * resolution <= inflate_radius + 1e-6:
                    offsets.append((dx, dy, dz))

    inflated = occupancy.copy()
    occupied_indices = np.argwhere(occupancy)
    shape = np.asarray(occupancy.shape, dtype=np.int32)

    for dx, dy, dz in offsets:
        shifted = occupied_indices + np.array([dx, dy, dz], dtype=np.int32)
        valid = np.all(shifted >= 0, axis=1) & np.all(shifted < shape[None, :], axis=1)
        shifted = shifted[valid]
        inflated[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = True

    return inflated


def build_search_offsets(max_radius_voxels: int) -> List[np.ndarray]:
    offsets: List[np.ndarray] = []
    for dx in range(-max_radius_voxels, max_radius_voxels + 1):
        for dy in range(-max_radius_voxels, max_radius_voxels + 1):
            for dz in range(-max_radius_voxels, max_radius_voxels + 1):
                offsets.append(np.array([dx, dy, dz], dtype=np.int32))
    offsets.sort(key=lambda delta: (np.linalg.norm(delta), abs(delta[2]), abs(delta[0]) + abs(delta[1])))
    return offsets


def is_in_bounds(index: np.ndarray, shape: Tuple[int, int, int]) -> bool:
    return bool(np.all(index >= 0) and np.all(index < np.asarray(shape, dtype=np.int32)))


def snap_center_to_free(
    occupancy: np.ndarray,
    center_index: np.ndarray,
    search_offsets: List[np.ndarray],
) -> Tuple[int, int, int]:
    if is_in_bounds(center_index, occupancy.shape) and not occupancy[tuple(center_index)]:
        return tuple(int(v) for v in center_index)

    for delta in search_offsets:
        candidate = center_index + delta
        if not is_in_bounds(candidate, occupancy.shape):
            continue
        if not occupancy[tuple(candidate)]:
            return tuple(int(v) for v in candidate)

    raise RuntimeError(f"Failed to find a free voxel near region center index {center_index.tolist()}")


NEIGHBORS: List[Tuple[int, int, int, float]] = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == 0 and dy == 0 and dz == 0:
                continue
            NEIGHBORS.append((dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz)))


def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return math.dist(a, b)


def reconstruct_path(
    parent_map: Dict[Tuple[int, int, int], Tuple[int, int, int]],
    current: Tuple[int, int, int],
) -> np.ndarray:
    path = [current]
    while current in parent_map:
        current = parent_map[current]
        path.append(current)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def a_star_3d(
    occupancy: np.ndarray,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    max_expansions: int,
    heuristic_weight: float = 1.0,
) -> np.ndarray:
    if occupancy[start] or occupancy[goal]:
        raise RuntimeError("Start or goal voxel is occupied after snapping.")

    sx, sy, sz = occupancy.shape
    open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = [
        (heuristic_weight * heuristic(start, goal), 0.0, start)
    ]
    parent_map: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], float] = {start: 0.0}
    closed: set[Tuple[int, int, int]] = set()
    expansions = 0

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1

        if current == goal:
            return reconstruct_path(parent_map, current)
        if expansions > max_expansions:
            raise RuntimeError(f"A* exceeded max expansions ({max_expansions}) from {start} to {goal}")

        cx, cy, cz = current
        for dx, dy, dz, step_cost in NEIGHBORS:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if not (0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz):
                continue
            if occupancy[nx, ny, nz]:
                continue

            neighbor = (nx, ny, nz)
            tentative_cost = current_cost + step_cost
            if tentative_cost >= g_score.get(neighbor, float("inf")):
                continue

            parent_map[neighbor] = current
            g_score[neighbor] = tentative_cost
            priority = tentative_cost + heuristic_weight * heuristic(neighbor, goal)
            heapq.heappush(open_heap, (priority, tentative_cost, neighbor))

    raise RuntimeError(f"No path found between {start} and {goal}")


def extract_local_subgrid(
    occupancy: np.ndarray,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    margin_voxels: int,
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int], np.ndarray]:
    start_np = np.asarray(start, dtype=np.int32)
    goal_np = np.asarray(goal, dtype=np.int32)
    shape = np.asarray(occupancy.shape, dtype=np.int32)

    min_corner = np.maximum(np.minimum(start_np, goal_np) - margin_voxels, 0)
    max_corner = np.minimum(np.maximum(start_np, goal_np) + margin_voxels + 1, shape)

    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(min_corner, max_corner))
    subgrid = occupancy[slices]
    local_start = tuple((start_np - min_corner).tolist())
    local_goal = tuple((goal_np - min_corner).tolist())
    return subgrid, local_start, local_goal, min_corner


def plan_path_with_adaptive_window(
    occupancy: np.ndarray,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    resolution: float,
    max_expansions: int,
    base_margin_m: float,
    margin_step_m: float,
    max_margin_m: float,
    heuristic_weight: float,
) -> np.ndarray:
    margins_m: List[float] = []
    current_margin = max(base_margin_m, resolution)
    while current_margin <= max_margin_m + 1e-6:
        margins_m.append(current_margin)
        current_margin += margin_step_m

    last_error: Optional[Exception] = None
    for margin_m in margins_m:
        margin_voxels = max(1, int(math.ceil(margin_m / resolution)))
        subgrid, local_start, local_goal, offset = extract_local_subgrid(
            occupancy=occupancy,
            start=start,
            goal=goal,
            margin_voxels=margin_voxels,
        )
        try:
            local_path = a_star_3d(
                occupancy=subgrid,
                start=local_start,
                goal=local_goal,
                max_expansions=max_expansions,
                heuristic_weight=heuristic_weight,
            )
            return local_path + offset[None, :]
        except RuntimeError as exc:
            last_error = exc
            continue

    if last_error is None:
        raise RuntimeError("Adaptive window planning failed without a concrete error.")
    raise RuntimeError(str(last_error))


def line_of_sight_is_free(
    occupancy: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> bool:
    delta = goal.astype(np.float32) - start.astype(np.float32)
    steps = int(np.ceil(np.linalg.norm(delta) * 2.0)) + 1
    for t in np.linspace(0.0, 1.0, max(steps, 2), dtype=np.float32):
        sample = np.rint(start + t * delta).astype(np.int32)
        if not is_in_bounds(sample, occupancy.shape):
            return False
        if occupancy[tuple(sample)]:
            return False
    return True


def shorten_path(path_indices: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    if len(path_indices) <= 2:
        return path_indices

    shortened = [path_indices[0]]
    cursor = 0
    while cursor < len(path_indices) - 1:
        next_idx = len(path_indices) - 1
        while next_idx > cursor + 1:
            if line_of_sight_is_free(occupancy, path_indices[cursor], path_indices[next_idx]):
                break
            next_idx -= 1
        shortened.append(path_indices[next_idx])
        cursor = next_idx

    return np.asarray(shortened, dtype=np.int32)


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) <= 1:
        return points.astype(np.float32)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length < 1e-6:
        return points[:1].astype(np.float32)

    sample_distances = np.arange(0.0, total_length, spacing, dtype=np.float32)
    if sample_distances.size == 0 or sample_distances[-1] < total_length:
        sample_distances = np.append(sample_distances, total_length)

    samples = []
    segment_index = 0
    for distance in sample_distances:
        while segment_index + 1 < len(cumulative) and cumulative[segment_index + 1] < distance:
            segment_index += 1
        if segment_index >= len(points) - 1:
            samples.append(points[-1])
            continue

        seg_start = points[segment_index]
        seg_end = points[segment_index + 1]
        seg_length = segment_lengths[segment_index]
        if seg_length < 1e-6:
            samples.append(seg_start)
            continue

        alpha = (distance - cumulative[segment_index]) / seg_length
        samples.append(seg_start + alpha * (seg_end - seg_start))

    return np.asarray(samples, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 3D voxel guidance paths from flysite point cloud.")
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=Path("flysite.ply"),
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=Path("DR_Surface_BBox_Data.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DR_Region_Guidance_Paths.npz"),
    )
    parser.add_argument("--resolution", type=float, default=0.1, help="Voxel resolution in meters.")
    parser.add_argument("--inflate-radius", type=float, default=0.25, help="Obstacle inflation radius in meters.")
    parser.add_argument("--padding", type=float, default=1.0, help="Padding around point cloud bounds in meters.")
    parser.add_argument("--resample-spacing", type=float, default=0.25, help="Uniform path resampling spacing in meters.")
    parser.add_argument(
        "--snap-radius",
        type=float,
        default=1.0,
        help="Radius for snapping region centers to nearest free voxel in meters.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=500000,
        help="Maximum number of A* expansions per path.",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=1.25,
        help="Weighted-A* heuristic multiplier. Values > 1 trade optimality for speed.",
    )
    parser.add_argument(
        "--local-margin",
        type=float,
        default=3.0,
        help="Initial local search margin around the start-goal bounding box in meters.",
    )
    parser.add_argument(
        "--local-margin-step",
        type=float,
        default=2.0,
        help="Margin growth step in meters when local planning fails.",
    )
    parser.add_argument(
        "--max-local-margin",
        type=float,
        default=12.0,
        help="Maximum local search margin in meters.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.point_cloud = resolve_project_path(args.point_cloud)
    args.bbox_file = resolve_project_path(args.bbox_file)
    args.output = resolve_project_path(args.output)
    t0 = time.time()

    print(f"Point cloud : {args.point_cloud}")
    print(f"BBox file   : {args.bbox_file}")
    print(f"Output NPZ  : {args.output}")
    print(f"Resolution  : {args.resolution:.3f} m")
    print(f"Inflation   : {args.inflate_radius:.3f} m")

    regions = parse_region_boxes(args.bbox_file)
    region_names = [region.name for region in regions]
    print(f"Loaded {len(regions)} regions.")

    points = load_binary_ply_vertices(args.point_cloud)
    print(f"Loaded point cloud vertices: {len(points):,}")

    occupancy, origin = build_occupancy_grid(
        points=points,
        regions=regions,
        resolution=args.resolution,
        padding=args.padding,
    )
    print(f"Occupancy grid shape before inflation: {occupancy.shape}")

    occupancy = inflate_occupancy(occupancy, inflate_radius=args.inflate_radius, resolution=args.resolution)
    occupied_ratio = float(occupancy.mean())
    print(f"Occupancy ratio after inflation: {occupied_ratio:.4f}")

    snap_radius_voxels = max(1, int(math.ceil(args.snap_radius / args.resolution)))
    search_offsets = build_search_offsets(snap_radius_voxels)

    region_center_points = np.stack([region.center for region in regions], axis=0)
    region_center_indices = world_to_grid(region_center_points, origin, args.resolution)
    snapped_centers: Dict[int, Tuple[int, int, int]] = {}
    for region, center_index in zip(regions, region_center_indices):
        snapped_centers[region.region_id] = snap_center_to_free(occupancy, center_index, search_offsets)

    directed_paths: Dict[Tuple[int, int], np.ndarray] = {}
    total_unique_pairs = len(regions) * (len(regions) - 1) // 2
    success_pairs = 0
    failed_pairs: List[str] = []
    pair_counter = 0

    for start_idx in range(len(regions)):
        for goal_idx in range(start_idx + 1, len(regions)):
            pair_counter += 1
            start_region = regions[start_idx]
            goal_region = regions[goal_idx]
            start_center = snapped_centers[start_region.region_id]
            goal_center = snapped_centers[goal_region.region_id]

            print(
                f"[{pair_counter}/{total_unique_pairs}] "
                f"Searching {start_region.region_id}:{start_region.name} <-> "
                f"{goal_region.region_id}:{goal_region.name}"
            )

            try:
                path_indices = plan_path_with_adaptive_window(
                    occupancy=occupancy,
                    start=start_center,
                    goal=goal_center,
                    resolution=args.resolution,
                    max_expansions=args.max_expansions,
                    base_margin_m=args.local_margin,
                    margin_step_m=args.local_margin_step,
                    max_margin_m=args.max_local_margin,
                    heuristic_weight=args.heuristic_weight,
                )
            except RuntimeError as exc:
                failure_desc = (
                    f"{start_region.region_id}:{start_region.name} <-> "
                    f"{goal_region.region_id}:{goal_region.name}: {exc}"
                )
                failed_pairs.append(failure_desc)
                print(f"[WARN] Failed to plan {failure_desc}")
                continue

            shortened_indices = shorten_path(path_indices, occupancy)
            path_world = grid_to_world(shortened_indices, origin, args.resolution)
            dense_path = resample_polyline(path_world, args.resample_spacing)

            directed_paths[(start_region.region_id, goal_region.region_id)] = dense_path.astype(np.float32)
            directed_paths[(goal_region.region_id, start_region.region_id)] = dense_path[::-1].astype(np.float32)
            success_pairs += 1

    save_region_guidance_paths(str(args.output), directed_paths, region_names=region_names)
    elapsed = time.time() - t0

    print(f"Saved directed paths to {args.output}")
    print(
        f"Computed {success_pairs}/{total_unique_pairs} unique region pairs "
        f"and wrote {len(directed_paths)} directed entries for compatibility."
    )
    if failed_pairs:
        print(f"Failed pairs: {len(failed_pairs)}")
        for failed_pair in failed_pairs[:20]:
            print(f"  - {failed_pair}")
        if len(failed_pairs) > 20:
            print(f"  ... and {len(failed_pairs) - 20} more")
    print(f"Elapsed time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
