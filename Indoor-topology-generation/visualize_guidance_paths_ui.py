#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Tuple

import numpy as np

from path_utils import resolve_project_path


@dataclass(frozen=True)
class Region:
    region_id: int
    name: str
    corners: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


def load_region_guidance_paths(input_path: str) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[str]]:
    loaded = np.load(input_path, allow_pickle=True)
    region_paths: Dict[Tuple[int, int], np.ndarray] = {}
    region_names: List[str] = []

    for key in loaded.files:
        if key == "region_names":
            region_names = [str(name) for name in loaded[key].tolist()]
            continue
        if not key.startswith("path_"):
            continue

        parts = key.split("_")
        if len(parts) != 3:
            continue

        start_region_id = int(parts[1])
        target_region_id = int(parts[2])
        region_paths[(start_region_id, target_region_id)] = np.asarray(loaded[key], dtype=np.float32)

    return region_paths, region_names


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


def canonicalize_paths(
    directed_paths: Dict[Tuple[int, int], np.ndarray],
) -> Dict[Tuple[int, int], np.ndarray]:
    canonical: Dict[Tuple[int, int], np.ndarray] = {}
    for (start_id, goal_id), path in directed_paths.items():
        if start_id == goal_id:
            continue
        pair = (min(start_id, goal_id), max(start_id, goal_id))
        if pair in canonical:
            continue
        if start_id <= goal_id:
            canonical[pair] = np.asarray(path, dtype=np.float32)
        else:
            canonical[pair] = np.asarray(path[::-1], dtype=np.float32)
    return canonical


def color_from_index(index: int, total: int) -> np.ndarray:
    hue = 0.0 if total <= 1 else float(index) / float(total)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return np.array([r, g, b], dtype=np.float64)


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
        color = [1.0, 0.2, 0.2] if region.region_id == selected_region_id else [1.0, 0.9, 0.1]
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
    colors = np.tile(np.array([[0.1, 0.9, 0.1]], dtype=np.float64), (len(regions), 1))
    colors[selected_region_id] = np.array([1.0, 0.1, 0.1], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_path_geometries(
    canonical_paths: Dict[Tuple[int, int], np.ndarray],
    selected_region_id: int,
) -> Tuple[object, object]:
    import open3d as o3d

    pair_items = [
        (pair, path)
        for pair, path in sorted(canonical_paths.items())
        if selected_region_id in pair
    ]
    line_points: List[List[float]] = []
    line_indices: List[List[int]] = []
    line_colors: List[List[float]] = []
    dense_points: List[List[float]] = []
    dense_colors: List[List[float]] = []

    for path_index, (pair, path) in enumerate(pair_items):
        color = color_from_index(path_index, len(pair_items))
        base_index = len(line_points)
        line_points.extend(path.astype(np.float64).tolist())
        dense_points.extend(path.astype(np.float64).tolist())
        dense_colors.extend([color.tolist()] * len(path))

        for point_index in range(len(path) - 1):
            line_indices.append([base_index + point_index, base_index + point_index + 1])
            line_colors.append(color.tolist())

    line_set = o3d.geometry.LineSet()
    if line_points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
    else:
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

    point_cloud = o3d.geometry.PointCloud()
    if dense_points:
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(dense_points, dtype=np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(dense_colors, dtype=np.float64))
    else:
        point_cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
    return line_set, point_cloud


def count_connected_paths(
    canonical_paths: Dict[Tuple[int, int], np.ndarray],
    selected_region_id: int,
) -> int:
    return sum(1 for pair in canonical_paths if selected_region_id in pair)


class PersistentPathViewer:
    def __init__(
        self,
        mesh_path: Path,
        regions: List[Region],
        canonical_paths: Dict[Tuple[int, int], np.ndarray],
        initial_region_id: int,
    ) -> None:
        self.mesh_path = mesh_path
        self.regions = regions
        self.canonical_paths = canonical_paths
        self.selected_region_id = initial_region_id

        import open3d as o3d

        self.o3d = o3d
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if self.mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.72, 0.72, 0.72])

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Guidance Paths 3D Viewer",
            width=1600,
            height=900,
            visible=True,
        )
        self.window_open = True

        self.region_boxes = build_region_boxes_line_set(regions, initial_region_id)
        self.region_centers = build_region_centers(regions, initial_region_id)
        self.path_lines, self.path_points = build_path_geometries(canonical_paths, initial_region_id)

        self.vis.add_geometry(self.mesh, reset_bounding_box=True)
        self.vis.add_geometry(self.region_boxes, reset_bounding_box=False)
        self.vis.add_geometry(self.region_centers, reset_bounding_box=False)
        self.vis.add_geometry(self.path_lines, reset_bounding_box=False)
        self.vis.add_geometry(self.path_points, reset_bounding_box=False)
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_region(self, selected_region_id: int) -> None:
        if not self.window_open:
            return

        self.selected_region_id = selected_region_id
        new_region_boxes = build_region_boxes_line_set(self.regions, selected_region_id)
        new_region_centers = build_region_centers(self.regions, selected_region_id)
        new_path_lines, new_path_points = build_path_geometries(self.canonical_paths, selected_region_id)

        self.vis.remove_geometry(self.region_boxes, reset_bounding_box=False)
        self.vis.remove_geometry(self.region_centers, reset_bounding_box=False)
        self.vis.remove_geometry(self.path_lines, reset_bounding_box=False)
        self.vis.remove_geometry(self.path_points, reset_bounding_box=False)

        self.region_boxes = new_region_boxes
        self.region_centers = new_region_centers
        self.path_lines = new_path_lines
        self.path_points = new_path_points

        self.vis.add_geometry(self.region_boxes, reset_bounding_box=False)
        self.vis.add_geometry(self.region_centers, reset_bounding_box=False)
        self.vis.add_geometry(self.path_lines, reset_bounding_box=False)
        self.vis.add_geometry(self.path_points, reset_bounding_box=False)
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


def visualize_paths(
    mesh_path: Path,
    regions: List[Region],
    canonical_paths: Dict[Tuple[int, int], np.ndarray],
    selected_region_id: int,
) -> None:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh from {mesh_path}")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.72, 0.72])

    region_boxes = build_region_boxes_line_set(regions, selected_region_id)
    region_centers = build_region_centers(regions, selected_region_id)
    path_lines, path_points = build_path_geometries(canonical_paths, selected_region_id)

    o3d.visualization.draw_geometries(
        [mesh, region_boxes, region_centers, path_lines, path_points],
        window_name=f"Guidance Paths For Region {selected_region_id}",
        width=1600,
        height=900,
    )


class PathViewerApp:
    def __init__(
        self,
        root: tk.Tk,
        mesh_path: Path,
        regions: List[Region],
        canonical_paths: Dict[Tuple[int, int], np.ndarray],
    ) -> None:
        self.root = root
        self.mesh_path = mesh_path
        self.regions = regions
        self.canonical_paths = canonical_paths
        self.viewer = PersistentPathViewer(
            mesh_path=mesh_path,
            regions=regions,
            canonical_paths=canonical_paths,
            initial_region_id=0,
        )

        self.root.title("Guidance Path Viewer")
        self.region_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar()

        info_text = f"Region ID range: 0 - {len(regions) - 1}"
        ttk.Label(root, text=info_text).pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(root, text="3D window opens once and stays alive. Switch region here to refresh the displayed paths.").pack(
            anchor="w", padx=12, pady=(0, 8)
        )

        input_frame = ttk.Frame(root)
        input_frame.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(input_frame, text="Region ID:").pack(side="left")
        self.entry = ttk.Entry(input_frame, textvariable=self.region_var, width=8)
        self.entry.pack(side="left", padx=(8, 8))
        self.entry.bind("<Return>", self.on_show_paths)
        ttk.Button(input_frame, text="Apply", command=self.on_show_paths).pack(side="left")
        ttk.Button(input_frame, text="Prev", command=self.on_prev_region).pack(side="left", padx=(8, 4))
        ttk.Button(input_frame, text="Next", command=self.on_next_region).pack(side="left")

        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", padx=12, pady=(0, 8))

        self.listbox = tk.Listbox(root, height=min(len(regions), 24), width=48)
        for region in regions:
            self.listbox.insert(tk.END, f"{region.region_id:02d}  {region.name}")
        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self.listbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.apply_region_id(0)
        self.schedule_viewer_tick()

    def on_listbox_select(self, _event: object) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        region_id = int(selection[0])
        self.region_var.set(str(region_id))
        self.apply_region_id(region_id)

    def on_show_paths(self, _event: object | None = None) -> None:
        try:
            region_id = int(self.region_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Region ID must be an integer.")
            return

        if region_id < 0 or region_id >= len(self.regions):
            messagebox.showerror(
                "Invalid Region ID",
                f"Region ID must be in range 0 - {len(self.regions) - 1}.",
            )
            return

        try:
            self.apply_region_id(region_id)
        except Exception as exc:  # pragma: no cover - UI error surface
            messagebox.showerror("Visualization Error", str(exc))

    def on_prev_region(self) -> None:
        current = int(self.region_var.get().strip() or "0")
        self.apply_region_id(max(0, current - 1))

    def on_next_region(self) -> None:
        current = int(self.region_var.get().strip() or "0")
        self.apply_region_id(min(len(self.regions) - 1, current + 1))

    def apply_region_id(self, region_id: int) -> None:
        self.region_var.set(str(region_id))
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(region_id)
        self.listbox.activate(region_id)
        self.listbox.see(region_id)
        self.viewer.update_region(region_id)

        num_paths = count_connected_paths(self.canonical_paths, region_id)
        region_name = self.regions[region_id].name
        self.status_var.set(f"Current region: {region_id}  {region_name} | displayed paths: {num_paths}")

    def schedule_viewer_tick(self) -> None:
        if self.viewer.tick():
            self.root.after(30, self.schedule_viewer_tick)
            return
        self.on_close()

    def on_close(self) -> None:
        self.viewer.close()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize region guidance paths with a simple UI.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("knife.ply"),
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=Path("DR_Surface_BBox_Data.txt"),
    )
    parser.add_argument(
        "--guidance-file",
        type=Path,
        default=Path("DR_Region_Guidance_Paths.npz"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.mesh = resolve_project_path(args.mesh)
    args.bbox_file = resolve_project_path(args.bbox_file)
    args.guidance_file = resolve_project_path(args.guidance_file)

    regions = parse_region_boxes(args.bbox_file)
    directed_paths, region_names = load_region_guidance_paths(str(args.guidance_file))
    if region_names and len(region_names) == len(regions):
        for region, region_name in zip(regions, region_names):
            if region.name != region_name:
                print(f"[WARN] Region name mismatch: bbox={region.name}, guidance={region_name}")

    canonical_paths = canonicalize_paths(directed_paths)
    print(f"Loaded {len(regions)} regions.")
    print(f"Loaded {len(directed_paths)} directed paths and {len(canonical_paths)} undirected paths.")
    print(f"Region ID range: 0 - {len(regions) - 1}")

    root = tk.Tk()
    PathViewerApp(root, mesh_path=args.mesh, regions=regions, canonical_paths=canonical_paths)
    root.mainloop()


if __name__ == "__main__":
    main()
