#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Tuple

import json
import numpy as np

from path_utils import PROJECT_DIR, resolve_project_path


@dataclass(frozen=True)
class Region:
    region_id: int
    name: str
    corners: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


@dataclass(frozen=True)
class PairResult:
    pair_index: int
    source_id: int
    source_name: str
    target_id: int
    target_name: str
    collision_free: bool
    num_path_points_checked: int
    num_collided_points_history: int
    num_collided_points_latest: int
    max_history_contact_force_norm: float
    max_latest_contact_force_norm: float
    points_file: Path

    @property
    def pair_key(self) -> Tuple[int, int]:
        return (self.source_id, self.target_id)

    def contains_region(self, region_id: int) -> bool:
        return self.source_id == region_id or self.target_id == region_id

    def other_region_id(self, region_id: int) -> int:
        if self.source_id == region_id:
            return self.target_id
        if self.target_id == region_id:
            return self.source_id
        raise ValueError(f"Region {region_id} not in pair {self.pair_key}")

    def other_region_name(self, region_id: int) -> str:
        if self.source_id == region_id:
            return self.target_name
        if self.target_id == region_id:
            return self.source_name
        raise ValueError(f"Region {region_id} not in pair {self.pair_key}")


def parse_region_boxes(bbox_path: Path) -> List[Region]:
    regions: List[Region] = []
    current_name = None
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
            x = float(tokens[tokens.index("X:") + 1])
            y = float(tokens[tokens.index("Y:") + 1])
            z = float(tokens[tokens.index("Z:") + 1])
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


def resolve_default_mesh() -> Path:
    exact_mesh = PROJECT_DIR / "DR_static_mesh_collision.ply"
    if exact_mesh.exists():
        return Path("DR_static_mesh_collision.ply")
    return Path("flysite.ply")


def load_contact_report(report_path: Path) -> Tuple[List[PairResult], Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results: List[PairResult] = []
    report_dir = report_path.parent
    raw_points_dir = Path(payload.get("points_dir", "")) if payload.get("points_dir") else None
    if raw_points_dir is None:
        points_dir = report_dir
    elif raw_points_dir.is_absolute():
        points_dir = raw_points_dir
    else:
        points_dir = report_dir / raw_points_dir

    for raw in payload.get("results", []):
        points_file = Path(raw["points_file"])
        if not points_file.is_absolute():
            candidates = [
                report_dir / points_file,
                points_dir / points_file,
                points_dir / points_file.name,
                report_dir / points_file.name,
                PROJECT_DIR / points_file,
            ]
            points_file = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        results.append(
            PairResult(
                pair_index=int(raw["pair_index"]),
                source_id=int(raw["source_id"]),
                source_name=str(raw["source_name"]),
                target_id=int(raw["target_id"]),
                target_name=str(raw["target_name"]),
                collision_free=bool(raw["collision_free"]),
                num_path_points_checked=int(raw["num_path_points_checked"]),
                num_collided_points_history=int(raw["num_collided_points_history"]),
                num_collided_points_latest=int(raw["num_collided_points_latest"]),
                max_history_contact_force_norm=float(raw["max_history_contact_force_norm"]),
                max_latest_contact_force_norm=float(raw["max_latest_contact_force_norm"]),
                points_file=points_file,
            )
        )

    if not results:
        raise RuntimeError(f"No pair results found in {report_path}")

    return results, points_dir


def load_safe_points_file(safe_points_path: Path | None) -> Dict[int, np.ndarray]:
    if safe_points_path is None or not safe_points_path.exists():
        return {}

    payload = np.load(safe_points_path)
    region_starts = np.asarray(payload["region_start_indices"], dtype=np.int64)
    region_counts = np.asarray(payload["region_counts"], dtype=np.int64)
    points_xyz = np.asarray(payload["points_xyz"], dtype=np.float32)

    safe_points_by_region: Dict[int, np.ndarray] = {}
    for region_id, (start, count) in enumerate(zip(region_starts.tolist(), region_counts.tolist(), strict=True)):
        safe_points_by_region[region_id] = points_xyz[start : start + count].astype(np.float64, copy=False)
    return safe_points_by_region


def connected_results(results: List[PairResult], region_id: int) -> List[PairResult]:
    return sorted(
        [result for result in results if result.contains_region(region_id)],
        key=lambda result: (result.other_region_id(region_id), result.source_id, result.target_id),
    )


def build_region_boxes_line_set(regions: List[Region], selected_region_id: int):
    import open3d as o3d

    points: List[List[float]] = []
    lines: List[List[int]] = []
    colors: List[List[float]] = []
    box_edges = [(0, 1), (1, 3), (3, 2), (2, 0)]

    for region in regions:
        corners = region.corners
        base_index = len(points)
        points.extend(corners.tolist())
        color = [1.0, 0.25, 0.25] if region.region_id == selected_region_id else [0.95, 0.85, 0.15]
        for edge_start, edge_end in box_edges:
            lines.append([base_index + edge_start, base_index + edge_end])
            colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def build_region_centers(regions: List[Region], selected_region_id: int):
    import open3d as o3d

    points = np.stack([region.center for region in regions], axis=0).astype(np.float64)
    colors = np.tile(np.array([[0.85, 0.95, 0.2]], dtype=np.float64), (len(regions), 1))
    colors[selected_region_id] = np.array([1.0, 0.2, 0.2], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_mesh_wireframe(mesh_path: Path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh from {mesh_path}")

    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    edge_set: set[Tuple[int, int]] = set()
    for triangle in triangles:
        i0, i1, i2 = [int(v) for v in triangle]
        edge_set.add(tuple(sorted((i0, i1))))
        edge_set.add(tuple(sorted((i1, i2))))
        edge_set.add(tuple(sorted((i2, i0))))

    edge_array = np.asarray(sorted(edge_set), dtype=np.int32)
    colors = np.tile(np.array([[0.45, 0.45, 0.45]], dtype=np.float64), (len(edge_array), 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edge_array)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def build_multi_path_geometries(path_items: List[Tuple[np.ndarray, np.ndarray]]):
    import open3d as o3d

    line_set = o3d.geometry.LineSet()
    point_cloud = o3d.geometry.PointCloud()
    collision_cloud = o3d.geometry.PointCloud()
    all_points: List[List[float]] = []
    all_lines: List[List[int]] = []
    all_line_colors: List[List[float]] = []
    all_point_colors: List[List[float]] = []
    collision_points: List[List[float]] = []

    for path_xyz_raw, collided_mask_raw in path_items:
        path_xyz = np.asarray(path_xyz_raw, dtype=np.float64)
        collided_mask = np.asarray(collided_mask_raw, dtype=bool)
        if path_xyz.ndim != 2 or path_xyz.shape[0] == 0 or path_xyz.shape[1] != 3:
            continue
        base_index = len(all_points)
        all_points.extend(path_xyz.tolist())

        point_colors = np.tile(np.array([[0.05, 0.85, 0.18]], dtype=np.float64), (len(path_xyz), 1))
        point_colors[collided_mask] = np.array([1.0, 0.08, 0.08], dtype=np.float64)
        all_point_colors.extend(point_colors.tolist())

        if len(path_xyz) > 1:
            local_lines = np.column_stack([np.arange(len(path_xyz) - 1), np.arange(1, len(path_xyz))]).astype(np.int32)
            local_lines += base_index
            all_lines.extend(local_lines.tolist())
            segment_collision = collided_mask[:-1] | collided_mask[1:]
            line_colors = np.tile(np.array([[0.05, 0.85, 0.18]], dtype=np.float64), (len(local_lines), 1))
            line_colors[segment_collision] = np.array([1.0, 0.08, 0.08], dtype=np.float64)
            all_line_colors.extend(line_colors.tolist())

        if np.any(collided_mask):
            collision_points.extend(path_xyz[collided_mask].tolist())

    if not all_points:
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        point_cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        collision_cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        collision_cloud.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        return line_set, point_cloud, collision_cloud

    line_set.points = o3d.utility.Vector3dVector(np.asarray(all_points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(all_lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(all_line_colors, dtype=np.float64))

    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(all_points, dtype=np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(all_point_colors, dtype=np.float64))

    collision_colors = np.tile(np.array([[1.0, 0.08, 0.08]], dtype=np.float64), (len(collision_points), 1))
    collision_cloud.points = o3d.utility.Vector3dVector(np.asarray(collision_points, dtype=np.float64))
    collision_cloud.colors = o3d.utility.Vector3dVector(collision_colors)
    return line_set, point_cloud, collision_cloud


def build_safe_points_cloud(points_xyz: np.ndarray):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        cloud.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        return cloud

    colors = np.tile(np.array([[0.1, 0.65, 1.0]], dtype=np.float64), (len(points), 1))
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


class PersistentContactViewer:
    def __init__(self, mesh_path: Path, regions: List[Region]) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.regions = regions
        self.mesh_wireframe = build_mesh_wireframe(mesh_path)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Fixed Height Contact Viewer", width=1700, height=950, visible=True)
        self.window_open = True

        self.region_boxes = None
        self.region_centers = None
        self.path_lines = None
        self.path_points = None
        self.collision_points = None
        self.safe_points = None

        self.vis.add_geometry(self.mesh_wireframe, reset_bounding_box=True)

    def update_selection(
        self,
        region_id: int,
        path_items: List[Tuple[np.ndarray, np.ndarray]],
        safe_points_xyz: np.ndarray | None,
    ) -> None:
        if not self.window_open:
            return

        new_region_boxes = build_region_boxes_line_set(self.regions, region_id)
        new_region_centers = build_region_centers(self.regions, region_id)
        new_path_lines, new_path_points, new_collision_points = build_multi_path_geometries(path_items)
        new_safe_points = build_safe_points_cloud(np.zeros((0, 3), dtype=np.float64) if safe_points_xyz is None else safe_points_xyz)

        for geometry in [
            self.region_boxes,
            self.region_centers,
            self.path_lines,
            self.path_points,
            self.collision_points,
            self.safe_points,
        ]:
            if geometry is not None:
                self.vis.remove_geometry(geometry, reset_bounding_box=False)

        self.region_boxes = new_region_boxes
        self.region_centers = new_region_centers
        self.path_lines = new_path_lines
        self.path_points = new_path_points
        self.collision_points = new_collision_points
        self.safe_points = new_safe_points

        for geometry in [
            self.region_boxes,
            self.region_centers,
            self.path_lines,
            self.path_points,
            self.collision_points,
            self.safe_points,
        ]:
            self.vis.add_geometry(geometry, reset_bounding_box=False)

        render_option = self.vis.get_render_option()
        render_option.line_width = 1.0
        render_option.point_size = 4.0
        self.tick()

    def tick(self) -> bool:
        if not self.window_open:
            return False
        self.window_open = bool(self.vis.poll_events())
        if self.window_open:
            self.vis.update_renderer()
        return self.window_open

    def close(self) -> None:
        if self.window_open:
            self.vis.destroy_window()
            self.window_open = False


class FixedHeightContactApp:
    def __init__(
        self,
        root: tk.Tk,
        mesh_path: Path,
        regions: List[Region],
        results: List[PairResult],
        safe_points_by_region: Dict[int, np.ndarray],
    ) -> None:
        self.root = root
        self.mesh_path = mesh_path
        self.regions = regions
        self.results = results
        self.safe_points_by_region = safe_points_by_region
        self.viewer = PersistentContactViewer(mesh_path=mesh_path, regions=regions)

        self.region_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="")
        self.show_safe_points_var = tk.BooleanVar(value=False)

        self.root.title("Fixed Height Contact Viewer")

        ttk.Label(root, text=f"Mesh: {mesh_path}").pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(
            root,
            text="选中一个区域后，显示该区域相关的全部轨迹。绿色为无碰撞段，红色为碰撞点/碰撞段，可选叠加当前区域的安全点。",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        region_frame = ttk.Frame(root)
        region_frame.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(region_frame, text="Region ID:").pack(side="left")
        self.region_entry = ttk.Entry(region_frame, textvariable=self.region_var, width=8)
        self.region_entry.pack(side="left", padx=(8, 8))
        self.region_entry.bind("<Return>", self.on_apply_region)
        ttk.Button(region_frame, text="Apply", command=self.on_apply_region).pack(side="left")
        ttk.Button(region_frame, text="Prev", command=self.on_prev_region).pack(side="left", padx=(8, 4))
        ttk.Button(region_frame, text="Next", command=self.on_next_region).pack(side="left")
        ttk.Checkbutton(
            region_frame,
            text="Show Safe Points",
            variable=self.show_safe_points_var,
            command=self.on_toggle_safe_points,
        ).pack(side="left", padx=(12, 0))

        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", padx=12, pady=(0, 8))

        self.listbox = tk.Listbox(root, height=min(len(regions), 24), width=56)
        for region in regions:
            self.listbox.insert(tk.END, f"{region.region_id:02d}  {region.name}")
        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self.listbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.listbox.bind("<<ListboxSelect>>", self.on_region_list_select)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.apply_region_id(0)
        self.schedule_viewer_tick()

    def on_region_list_select(self, _event: object) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        region_id = int(selection[0])
        self.region_var.set(str(region_id))
        self.apply_region_id(region_id)

    def on_apply_region(self, _event: object | None = None) -> None:
        try:
            region_id = int(self.region_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Region ID must be an integer.")
            return

        if region_id < 0 or region_id >= len(self.regions):
            messagebox.showerror("Invalid Region ID", f"Region ID must be in range 0 - {len(self.regions) - 1}.")
            return
        self.apply_region_id(region_id)

    def on_prev_region(self) -> None:
        current = int(self.region_var.get().strip() or "0")
        self.apply_region_id(max(0, current - 1))

    def on_next_region(self) -> None:
        current = int(self.region_var.get().strip() or "0")
        self.apply_region_id(min(len(self.regions) - 1, current + 1))

    def on_toggle_safe_points(self) -> None:
        current = int(self.region_var.get().strip() or "0")
        self.apply_viewer_selection(current)

    def apply_viewer_selection(self, region_id: int) -> None:
        connected = connected_results(self.results, region_id)
        safe_points_region = self.safe_points_by_region.get(region_id, np.zeros((0, 3), dtype=np.float64))
        safe_points_for_vis = safe_points_region if self.show_safe_points_var.get() else None
        if not connected:
            self.status_var.set(
                f"No connected paths for this region. safe_points={len(safe_points_region)}"
            )
            self.viewer.update_selection(region_id=region_id, path_items=[], safe_points_xyz=safe_points_for_vis)
            return

        path_items: List[Tuple[np.ndarray, np.ndarray]] = []
        collided_pairs = 0
        collision_free_pairs = 0
        total_collided_points = 0
        max_force = 0.0
        for result in connected:
            point_data = np.load(result.points_file)
            path_xyz = np.asarray(point_data["path_xyz"], dtype=np.float64)
            collided_mask = np.asarray(point_data["collided_history"], dtype=bool)
            path_items.append((path_xyz, collided_mask))
            total_collided_points += int(np.count_nonzero(collided_mask))
            max_force = max(max_force, float(result.max_history_contact_force_norm))
            if result.collision_free:
                collision_free_pairs += 1
            else:
                collided_pairs += 1

        self.viewer.update_selection(region_id=region_id, path_items=path_items, safe_points_xyz=safe_points_for_vis)
        self.status_var.set(
            f"Region {region_id} {self.regions[region_id].name}"
            f" | connected_pairs={len(connected)}"
            f" | collision_free_pairs={collision_free_pairs}"
            f" | collided_pairs={collided_pairs}"
            f" | total_collided_points={total_collided_points}"
            f" | safe_points={len(safe_points_region)}"
            f" | maxF(history)={max_force:.3f}"
        )

    def apply_region_id(self, region_id: int) -> None:
        self.region_var.set(str(region_id))
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(region_id)
        self.listbox.activate(region_id)
        self.listbox.see(region_id)
        self.apply_viewer_selection(region_id)

    def schedule_viewer_tick(self) -> None:
        if self.viewer.tick():
            self.root.after(30, self.schedule_viewer_tick)
            return
        self.on_close()

    def on_close(self) -> None:
        self.viewer.close()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize fixed-height contact results with Open3D.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=resolve_default_mesh(),
        help="PLY mesh used as wireframe background. Prefer the exported collision mesh.",
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=Path("DR_Surface_BBox_Data.txt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/fixed_height_validation/fixed_height_contact_full.json"),
    )
    parser.add_argument(
        "--safe-points",
        type=Path,
        default=Path("DR_region_safe_points_contact_0p2m_1p2_to_2p0_eroded_0p4m.npz"),
        help="Optional region-wise safe-point pool NPZ generated by generate_region_safe_points_contact.py.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.mesh = resolve_project_path(args.mesh)
    args.bbox_file = resolve_project_path(args.bbox_file)
    args.report = resolve_project_path(args.report)
    args.safe_points = resolve_project_path(args.safe_points)
    regions = parse_region_boxes(args.bbox_file)
    results, _ = load_contact_report(args.report)
    safe_points_by_region = load_safe_points_file(args.safe_points)
    print(f"Loaded {len(regions)} regions.")
    print(f"Loaded {len(results)} pair results.")
    print(f"Mesh: {args.mesh}")
    if args.safe_points and args.safe_points.exists():
        print(f"Safe points: {args.safe_points}")

    root = tk.Tk()
    FixedHeightContactApp(
        root=root,
        mesh_path=args.mesh,
        regions=regions,
        results=results,
        safe_points_by_region=safe_points_by_region,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
