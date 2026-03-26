#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, Iterable, List, Tuple

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


@dataclass(frozen=True)
class CorridorPolyhedron:
    halfspaces: np.ndarray
    vertices: np.ndarray


@dataclass(frozen=True)
class TrajectoryRecord:
    source_id: int
    source_name: str
    target_id: int
    target_name: str
    path_reachable: bool
    optimization_succeeded: bool
    failure_reason: str
    guide_route: np.ndarray
    resampled_path: np.ndarray
    breakpoints: np.ndarray
    coefficients: np.ndarray
    num_segments: int
    num_coefficients: int
    safe_corridor: List[CorridorPolyhedron]

    @property
    def pair_key(self) -> Tuple[int, int]:
        return (self.source_id, self.target_id)

    @property
    def undirected_key(self) -> Tuple[int, int]:
        return (min(self.source_id, self.target_id), max(self.source_id, self.target_id))

    def contains_region(self, region_id: int) -> bool:
        return self.source_id == region_id or self.target_id == region_id

    def other_region_id(self, region_id: int) -> int:
        if self.source_id == region_id:
            return self.target_id
        if self.target_id == region_id:
            return self.source_id
        raise ValueError(f"Region {region_id} not in trajectory {self.pair_key}")

    def other_region_name(self, region_id: int) -> str:
        if self.source_id == region_id:
            return self.target_name
        if self.target_id == region_id:
            return self.source_name
        raise ValueError(f"Region {region_id} not in trajectory {self.pair_key}")


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


def load_trajectory_records(trajectory_path: Path) -> List[TrajectoryRecord]:
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    raw_records = payload.get("trajectories", [])
    records: List[TrajectoryRecord] = []

    for raw in raw_records:
        safe_corridor = [
            CorridorPolyhedron(
                halfspaces=np.asarray(polyhedron.get("halfspaces", []), dtype=np.float64),
                vertices=np.asarray(polyhedron.get("vertices", []), dtype=np.float64),
            )
            for polyhedron in raw.get("safe_corridor", [])
        ]
        records.append(
            TrajectoryRecord(
                source_id=int(raw["source_id"]),
                source_name=str(raw["source_name"]),
                target_id=int(raw["target_id"]),
                target_name=str(raw["target_name"]),
                path_reachable=bool(raw.get("path_reachable", False)),
                optimization_succeeded=bool(raw.get("optimization_succeeded", False)),
                failure_reason=str(raw.get("failure_reason", "")),
                guide_route=np.asarray(raw.get("guide_route", []), dtype=np.float64),
                resampled_path=np.asarray(raw.get("resampled_path", []), dtype=np.float64),
                breakpoints=np.asarray(raw.get("breakpoints", []), dtype=np.float64),
                coefficients=np.asarray(raw.get("coefficients", []), dtype=np.float64),
                num_segments=int(raw.get("num_segments", 0)),
                num_coefficients=int(raw.get("num_coefficients", 0)),
                safe_corridor=safe_corridor,
            )
        )

    return records


def sample_trajectory(record: TrajectoryRecord, samples_per_segment: int) -> np.ndarray:
    if (
        not record.optimization_succeeded
        or record.num_segments <= 0
        or record.num_coefficients <= 0
        or record.breakpoints.size != record.num_segments + 1
        or record.coefficients.shape[0] != record.num_segments * record.num_coefficients
        or record.coefficients.shape[1] != 3
    ):
        return np.zeros((0, 3), dtype=np.float64)

    samples: List[np.ndarray] = []
    for segment_idx in range(record.num_segments):
        block_start = segment_idx * record.num_coefficients
        block_end = block_start + record.num_coefficients
        coeffs = record.coefficients[block_start:block_end]

        segment_duration = float(record.breakpoints[segment_idx + 1] - record.breakpoints[segment_idx])
        if segment_duration <= 0.0:
            continue

        local_times = np.linspace(
            0.0,
            segment_duration,
            max(samples_per_segment, 2),
            endpoint=False,
            dtype=np.float64,
        )
        if segment_idx == record.num_segments - 1:
            local_times = np.append(local_times, segment_duration)

        for dt in local_times:
            powers = np.power(dt, np.arange(record.num_coefficients, dtype=np.float64))
            samples.append((powers[:, None] * coeffs).sum(axis=0))

    if not samples:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def color_from_index(index: int, total: int) -> np.ndarray:
    hue = 0.0 if total <= 1 else float(index) / float(total)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return np.array([r, g, b], dtype=np.float64)


def darken_color(color: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(color * factor, 0.0, 1.0)


def brighten_color(color: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(1.0 - (1.0 - color) * factor, 0.0, 1.0)


def connected_trajectories(records: Iterable[TrajectoryRecord], region_id: int) -> List[TrajectoryRecord]:
    return sorted(
        [record for record in records if record.contains_region(region_id)],
        key=lambda record: (record.other_region_id(region_id), record.source_id, record.target_id),
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
    colors = np.tile(np.array([[0.15, 0.95, 0.15]], dtype=np.float64), (len(regions), 1))
    colors[selected_region_id] = np.array([1.0, 0.15, 0.15], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_polyline_geometries(
    polylines: List[np.ndarray],
    colors: List[np.ndarray],
) -> Tuple[object, object]:
    import open3d as o3d

    line_points: List[List[float]] = []
    line_indices: List[List[int]] = []
    line_colors: List[List[float]] = []
    dense_points: List[List[float]] = []
    dense_colors: List[List[float]] = []

    for polyline, color in zip(polylines, colors, strict=True):
        if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] != 3:
            continue
        base_index = len(line_points)
        polyline_list = polyline.astype(np.float64).tolist()
        line_points.extend(polyline_list)
        dense_points.extend(polyline_list)
        dense_colors.extend([color.tolist()] * len(polyline_list))
        for point_index in range(len(polyline_list) - 1):
            line_indices.append([base_index + point_index, base_index + point_index + 1])
            line_colors.append(color.tolist())

    line_set = o3d.geometry.LineSet()
    point_cloud = o3d.geometry.PointCloud()

    if line_points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(dense_points, dtype=np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(dense_colors, dtype=np.float64))
    else:
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        point_cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

    return line_set, point_cloud


def deduplicate_vertices(vertices: np.ndarray, tolerance: float) -> np.ndarray:
    unique: List[np.ndarray] = []
    for vertex in vertices:
        if not unique:
            unique.append(vertex)
            continue
        if any(np.linalg.norm(vertex - existing) <= tolerance for existing in unique):
            continue
        unique.append(vertex)
    if not unique:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(unique, dtype=np.float64)


def build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = np.linalg.norm(normal)
    if norm <= 1.0e-12:
        raise ValueError("Degenerate plane normal.")
    unit_normal = normal / norm

    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(unit_normal[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tangent = helper - np.dot(helper, unit_normal) * unit_normal
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(unit_normal, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)
    return unit_normal, tangent, bitangent


def order_face_vertices(vertices: np.ndarray, face_indices: np.ndarray, normal: np.ndarray) -> List[int]:
    face_vertices = vertices[face_indices]
    center = face_vertices.mean(axis=0)
    unit_normal, tangent, bitangent = build_plane_basis(normal)

    angles: List[Tuple[float, int]] = []
    for local_index, point in enumerate(face_vertices):
        offset = point - center
        x_coord = float(np.dot(offset, tangent))
        y_coord = float(np.dot(offset, bitangent))
        angles.append((math.atan2(y_coord, x_coord), local_index))

    angles.sort(key=lambda item: item[0])
    ordered = [int(face_indices[local_index]) for _, local_index in angles]

    for start in range(len(ordered) - 2):
        p0 = vertices[ordered[start]]
        p1 = vertices[ordered[start + 1]]
        p2 = vertices[ordered[start + 2]]
        face_normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(face_normal) <= 1.0e-12:
            continue
        if float(np.dot(face_normal, unit_normal)) < 0.0:
            ordered.reverse()
        break

    return ordered


def triangulate_polyhedron(vertices: np.ndarray, halfspaces: np.ndarray, plane_eps: float) -> List[Tuple[int, int, int]]:
    faces: List[Tuple[int, int, int]] = []
    seen_faces: set[Tuple[int, ...]] = set()

    for halfspace in halfspaces:
        normal = halfspace[:3]
        offset = float(halfspace[3])
        signed_distance = vertices @ normal + offset
        face_indices = np.flatnonzero(np.abs(signed_distance) <= plane_eps)
        if face_indices.size < 3:
            continue

        key = tuple(sorted(int(index) for index in face_indices.tolist()))
        if key in seen_faces:
            continue

        ordered = order_face_vertices(vertices, face_indices, normal)
        if len(ordered) < 3:
            continue

        for triangle_index in range(1, len(ordered) - 1):
            faces.append((ordered[0], ordered[triangle_index], ordered[triangle_index + 1]))
        seen_faces.add(key)

    return faces


def build_corridor_wireframe(
    record: TrajectoryRecord,
    plane_eps: float,
    dedup_eps: float,
):
    import open3d as o3d

    points: List[List[float]] = []
    edges: List[List[int]] = []
    colors: List[List[float]] = []

    for polyhedron in record.safe_corridor:
        if polyhedron.vertices.ndim != 2 or polyhedron.vertices.shape[0] < 4:
            continue
        if polyhedron.halfspaces.ndim != 2 or polyhedron.halfspaces.shape[0] < 4:
            continue

        vertices = deduplicate_vertices(polyhedron.vertices, dedup_eps)
        if vertices.shape[0] < 4:
            continue
        faces = triangulate_polyhedron(vertices, polyhedron.halfspaces, plane_eps)
        if not faces:
            continue

        base_index = len(points)
        points.extend(vertices.tolist())

        unique_edges: set[Tuple[int, int]] = set()
        for i0, i1, i2 in faces:
            for edge in ((i0, i1), (i1, i2), (i2, i0)):
                unique_edges.add(tuple(sorted(edge)))
        for edge_start, edge_end in sorted(unique_edges):
            edges.append([base_index + edge_start, base_index + edge_end])
            colors.append([0.35, 0.95, 1.0])

    line_set = o3d.geometry.LineSet()
    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    else:
        line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
    return line_set


class PersistentGuidanceTrajectoryViewer:
    def __init__(
        self,
        mesh_path: Path,
        regions: List[Region],
        trajectory_records: List[TrajectoryRecord],
        samples_per_segment: int,
        plane_eps: float,
        dedup_eps: float,
    ) -> None:
        self.mesh_path = mesh_path
        self.regions = regions
        self.trajectory_records = trajectory_records
        self.samples_per_segment = samples_per_segment
        self.plane_eps = plane_eps
        self.dedup_eps = dedup_eps
        self.sampled_trajectories: Dict[Tuple[int, int], np.ndarray] = {
            record.pair_key: sample_trajectory(record, samples_per_segment)
            for record in trajectory_records
        }

        import open3d as o3d

        self.o3d = o3d
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if self.mesh.is_empty():
            raise RuntimeError(f"Failed to load mesh from {mesh_path}")
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.72, 0.72, 0.72])

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Guidance + Trajectory + Corridor Viewer",
            width=1700,
            height=950,
            visible=True,
        )
        self.window_open = True

        self.region_boxes = None
        self.region_centers = None
        self.all_guides = None
        self.all_guide_points = None
        self.all_trajs = None
        self.all_traj_points = None
        self.selected_guide = None
        self.selected_guide_points = None
        self.selected_traj = None
        self.selected_traj_points = None
        self.selected_corridor = None

        self.vis.add_geometry(self.mesh, reset_bounding_box=True)
        self.update_selection(region_id=0, selected_pair_key=None)

    def update_selection(self, region_id: int, selected_pair_key: Tuple[int, int] | None) -> None:
        if not self.window_open:
            return

        connected = connected_trajectories(self.trajectory_records, region_id)
        pair_colors = {
            record.pair_key: color_from_index(index, len(connected))
            for index, record in enumerate(connected)
        }

        all_guide_polylines = [record.guide_route for record in connected]
        all_guide_colors = [brighten_color(pair_colors[record.pair_key], 0.70) for record in connected]
        all_traj_polylines = [self.sampled_trajectories[record.pair_key] for record in connected]
        all_traj_colors = [darken_color(pair_colors[record.pair_key], 0.65) for record in connected]

        selected_record = None
        if selected_pair_key is not None:
            selected_record = next((record for record in connected if record.pair_key == selected_pair_key), None)

        new_region_boxes = build_region_boxes_line_set(self.regions, region_id)
        new_region_centers = build_region_centers(self.regions, region_id)
        new_all_guides, new_all_guide_points = build_polyline_geometries(all_guide_polylines, all_guide_colors)
        new_all_trajs, new_all_traj_points = build_polyline_geometries(all_traj_polylines, all_traj_colors)

        if selected_record is not None:
            new_selected_guide, new_selected_guide_points = build_polyline_geometries(
                [selected_record.guide_route],
                [np.array([1.0, 0.95, 0.2], dtype=np.float64)],
            )
            sampled = self.sampled_trajectories[selected_record.pair_key]
            new_selected_traj, new_selected_traj_points = build_polyline_geometries(
                [sampled],
                [np.array([1.0, 0.3, 0.2], dtype=np.float64)],
            )
            new_selected_corridor = build_corridor_wireframe(
                selected_record,
                plane_eps=self.plane_eps,
                dedup_eps=self.dedup_eps,
            )
        else:
            empty = np.zeros((0, 3), dtype=np.float64)
            new_selected_guide, new_selected_guide_points = build_polyline_geometries(
                [empty],
                [np.array([1.0, 0.95, 0.2], dtype=np.float64)],
            )
            new_selected_traj, new_selected_traj_points = build_polyline_geometries(
                [empty],
                [np.array([1.0, 0.3, 0.2], dtype=np.float64)],
            )
            new_selected_corridor = build_corridor_wireframe(
                TrajectoryRecord(
                    source_id=-1,
                    source_name="",
                    target_id=-1,
                    target_name="",
                    path_reachable=False,
                    optimization_succeeded=False,
                    failure_reason="",
                    guide_route=empty,
                    resampled_path=empty,
                    breakpoints=np.zeros((0,), dtype=np.float64),
                    coefficients=np.zeros((0, 3), dtype=np.float64),
                    num_segments=0,
                    num_coefficients=0,
                    safe_corridor=[],
                ),
                plane_eps=self.plane_eps,
                dedup_eps=self.dedup_eps,
            )

        old_geometries = [
            self.region_boxes,
            self.region_centers,
            self.all_guides,
            self.all_guide_points,
            self.all_trajs,
            self.all_traj_points,
            self.selected_guide,
            self.selected_guide_points,
            self.selected_traj,
            self.selected_traj_points,
            self.selected_corridor,
        ]
        for geometry in old_geometries:
            if geometry is not None:
                self.vis.remove_geometry(geometry, reset_bounding_box=False)

        self.region_boxes = new_region_boxes
        self.region_centers = new_region_centers
        self.all_guides = new_all_guides
        self.all_guide_points = new_all_guide_points
        self.all_trajs = new_all_trajs
        self.all_traj_points = new_all_traj_points
        self.selected_guide = new_selected_guide
        self.selected_guide_points = new_selected_guide_points
        self.selected_traj = new_selected_traj
        self.selected_traj_points = new_selected_traj_points
        self.selected_corridor = new_selected_corridor

        new_geometries = [
            self.region_boxes,
            self.region_centers,
            self.all_guides,
            self.all_guide_points,
            self.all_trajs,
            self.all_traj_points,
            self.selected_corridor,
            self.selected_guide,
            self.selected_guide_points,
            self.selected_traj,
            self.selected_traj_points,
        ]
        for geometry in new_geometries:
            self.vis.add_geometry(geometry, reset_bounding_box=False)

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


class GuidanceTrajectoryApp:
    def __init__(
        self,
        root: tk.Tk,
        mesh_path: Path,
        regions: List[Region],
        trajectory_records: List[TrajectoryRecord],
        samples_per_segment: int,
        plane_eps: float,
        dedup_eps: float,
    ) -> None:
        self.root = root
        self.regions = regions
        self.trajectory_records = trajectory_records
        self.viewer = PersistentGuidanceTrajectoryViewer(
            mesh_path=mesh_path,
            regions=regions,
            trajectory_records=trajectory_records,
            samples_per_segment=samples_per_segment,
            plane_eps=plane_eps,
            dedup_eps=dedup_eps,
        )

        self.region_var = tk.StringVar(value="0")
        self.path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.path_choices: Dict[str, Tuple[int, int]] = {}

        self.root.title("Guidance + Trajectory Viewer")

        ttk.Label(
            root,
            text=f"Region ID range: 0 - {len(regions) - 1}",
        ).pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(
            root,
            text="选中区域后会显示该区域相关的 guide path 和优化轨迹；再选一条具体路径，会额外高亮并显示其安全飞行走廊线框。",
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

        path_frame = ttk.Frame(root)
        path_frame.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(path_frame, text="Path:").pack(side="left")
        self.path_combo = ttk.Combobox(path_frame, textvariable=self.path_var, state="readonly", width=68)
        self.path_combo.pack(side="left", fill="x", expand=True, padx=(8, 8))
        self.path_combo.bind("<<ComboboxSelected>>", self.on_path_selected)
        ttk.Button(path_frame, text="Clear", command=self.on_clear_path).pack(side="left")

        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", padx=12, pady=(0, 8))

        self.listbox = tk.Listbox(root, height=min(len(regions), 24), width=52)
        for region in regions:
            self.listbox.insert(tk.END, f"{region.region_id:02d}  {region.name}")
        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self.listbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.listbox.bind("<<ListboxSelect>>", self.on_region_list_select)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.apply_region_id(0)
        self.schedule_viewer_tick()

    def format_path_label(self, region_id: int, record: TrajectoryRecord) -> str:
        other_id = record.other_region_id(region_id)
        other_name = record.other_region_name(region_id)
        traj_status = "traj-ok" if record.optimization_succeeded else "traj-fail"
        corridor_status = f"corridor:{len(record.safe_corridor)}"
        return f"{record.source_id}->{record.target_id} | other={other_id:02d} {other_name} | {traj_status} | {corridor_status}"

    def refresh_path_options(self, region_id: int) -> None:
        connected = connected_trajectories(self.trajectory_records, region_id)
        self.path_choices = {
            self.format_path_label(region_id, record): record.pair_key
            for record in connected
        }
        labels = list(self.path_choices.keys())
        self.path_combo["values"] = labels
        if labels:
            self.path_var.set(labels[0])
        else:
            self.path_var.set("")

    def get_selected_pair(self) -> Tuple[int, int] | None:
        label = self.path_var.get().strip()
        if not label:
            return None
        return self.path_choices.get(label)

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
            messagebox.showerror(
                "Invalid Region ID",
                f"Region ID must be in range 0 - {len(self.regions) - 1}.",
            )
            return

        self.apply_region_id(region_id)

    def on_prev_region(self) -> None:
        current = int(self.region_var.get().strip() or "0")
        self.apply_region_id(max(0, current - 1))

    def on_next_region(self) -> None:
        current = int(self.region_var.get().strip() or "0")
        self.apply_region_id(min(len(self.regions) - 1, current + 1))

    def on_path_selected(self, _event: object | None = None) -> None:
        region_id = int(self.region_var.get().strip() or "0")
        self.apply_viewer_selection(region_id)

    def on_clear_path(self) -> None:
        self.path_var.set("")
        region_id = int(self.region_var.get().strip() or "0")
        self.apply_viewer_selection(region_id)

    def apply_viewer_selection(self, region_id: int) -> None:
        selected_pair = self.get_selected_pair()
        self.viewer.update_selection(region_id=region_id, selected_pair_key=selected_pair)

        connected = connected_trajectories(self.trajectory_records, region_id)
        region_name = self.regions[region_id].name
        status = f"Current region: {region_id}  {region_name} | connected pairs: {len(connected)}"
        if selected_pair is not None:
            record = next((item for item in connected if item.pair_key == selected_pair), None)
            if record is not None:
                status += (
                    f" | selected: {record.source_id}->{record.target_id}"
                    f" | traj={'ok' if record.optimization_succeeded else 'fail'}"
                    f" | corridor polyhedra={len(record.safe_corridor)}"
                )
        else:
            status += " | selected: none"
        self.status_var.set(status)

    def apply_region_id(self, region_id: int) -> None:
        self.region_var.set(str(region_id))
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(region_id)
        self.listbox.activate(region_id)
        self.listbox.see(region_id)
        self.refresh_path_options(region_id)
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
    parser = argparse.ArgumentParser(
        description="Visualize region guidance paths, optimized trajectories, and selected safe corridors."
    )
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
        "--trajectory-file",
        type=Path,
        default=Path("region_paths_txt/all_region_pair_trajectories.json"),
    )
    parser.add_argument(
        "--samples-per-segment",
        type=int,
        default=40,
        help="Number of samples used to draw each optimized spline segment.",
    )
    parser.add_argument(
        "--plane-eps",
        type=float,
        default=1.0e-4,
        help="Tolerance for corridor face reconstruction from halfspaces.",
    )
    parser.add_argument(
        "--dedup-eps",
        type=float,
        default=1.0e-5,
        help="Tolerance for corridor vertex deduplication.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.mesh = resolve_project_path(args.mesh)
    args.bbox_file = resolve_project_path(args.bbox_file)
    args.trajectory_file = resolve_project_path(args.trajectory_file)

    regions = parse_region_boxes(args.bbox_file)
    trajectory_records = load_trajectory_records(args.trajectory_file)
    print(f"Loaded {len(regions)} regions.")
    print(f"Loaded {len(trajectory_records)} trajectory records.")
    print(f"Region ID range: 0 - {len(regions) - 1}")

    root = tk.Tk()
    GuidanceTrajectoryApp(
        root,
        mesh_path=args.mesh,
        regions=regions,
        trajectory_records=trajectory_records,
        samples_per_segment=max(args.samples_per_segment, 2),
        plane_eps=args.plane_eps,
        dedup_eps=args.dedup_eps,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
