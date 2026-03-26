#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from path_utils import resolve_project_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert safe-flight corridors from all_region_pair_trajectories.json to an ASCII PLY mesh."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("region_paths_txt/all_region_pair_trajectories.json"),
        help="Input trajectory JSON exported by voxel_guidance_ros2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PLY path. If omitted, a name will be generated next to the input JSON.",
    )
    parser.add_argument(
        "--source-id",
        type=int,
        default=None,
        help="Source region id to export. By default, exports all trajectories.",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=None,
        help="Target region id to export. By default, exports all trajectories.",
    )
    parser.add_argument(
        "--exact-direction",
        action="store_true",
        help="Only match source_id -> target_id exactly. By default, (a, b) also matches (b, a).",
    )
    parser.add_argument(
        "--plane-eps",
        type=float,
        default=1.0e-4,
        help="Tolerance used to decide whether a vertex lies on a corridor face plane.",
    )
    parser.add_argument(
        "--dedup-eps",
        type=float,
        default=1.0e-5,
        help="Tolerance used to deduplicate nearly identical polyhedron vertices.",
    )
    return parser


def pick_default_output(input_path: Path, source_id: int | None, target_id: int | None) -> Path:
    if source_id is None or target_id is None:
        return input_path.with_name("all_safe_corridors.ply")
    return input_path.with_name(f"safe_corridor_{source_id}_{target_id}.ply")


def color_from_index(index: int, total: int) -> tuple[int, int, int]:
    hue = 0.0 if total <= 1 else float(index) / float(total)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def load_trajectories(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise RuntimeError(f"No 'trajectories' array found in {input_path}")
    return trajectories


def matches_pair(
    trajectory: dict,
    source_id: int | None,
    target_id: int | None,
    exact_direction: bool,
) -> bool:
    if source_id is None or target_id is None:
        return True

    traj_source = int(trajectory.get("source_id", -1))
    traj_target = int(trajectory.get("target_id", -1))
    if exact_direction:
        return traj_source == source_id and traj_target == target_id
    return (traj_source == source_id and traj_target == target_id) or (
        traj_source == target_id and traj_target == source_id
    )


def deduplicate_vertices(vertices: np.ndarray, tolerance: float) -> np.ndarray:
    unique: list[np.ndarray] = []
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


def build_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = np.linalg.norm(normal)
    if norm <= 1.0e-12:
        raise ValueError("Degenerate plane normal.")
    unit_normal = normal / norm

    if abs(unit_normal[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tangent = helper - np.dot(helper, unit_normal) * unit_normal
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm <= 1.0e-12:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tangent = helper - np.dot(helper, unit_normal) * unit_normal
        tangent_norm = np.linalg.norm(tangent)
    tangent = tangent / tangent_norm
    bitangent = np.cross(unit_normal, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)
    return unit_normal, tangent, bitangent


def order_face_vertices(vertices: np.ndarray, face_indices: np.ndarray, normal: np.ndarray) -> list[int]:
    face_vertices = vertices[face_indices]
    center = face_vertices.mean(axis=0)
    unit_normal, tangent, bitangent = build_plane_basis(normal)

    angles: list[tuple[float, int]] = []
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


def triangulate_polyhedron(vertices: np.ndarray, halfspaces: np.ndarray, plane_eps: float) -> list[tuple[int, int, int]]:
    faces: list[tuple[int, int, int]] = []
    seen_faces: set[tuple[int, ...]] = set()

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


def iter_selected_corridors(
    trajectories: Iterable[dict],
    source_id: int | None,
    target_id: int | None,
    exact_direction: bool,
) -> list[dict]:
    selected = [
        trajectory
        for trajectory in trajectories
        if matches_pair(trajectory, source_id, target_id, exact_direction)
    ]
    if not selected:
        raise RuntimeError("No trajectories matched the requested region pair.")
    return selected


def corridor_to_mesh(
    selected_trajectories: list[dict],
    plane_eps: float,
    dedup_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_vertices: list[np.ndarray] = []
    all_faces: list[tuple[int, int, int]] = []
    all_colors: list[tuple[int, int, int]] = []

    total_polyhedra = sum(len(trajectory.get("safe_corridor", [])) for trajectory in selected_trajectories)
    color_cursor = 0
    skipped_polyhedra = 0

    for trajectory in selected_trajectories:
        for polyhedron in trajectory.get("safe_corridor", []):
            raw_vertices = np.asarray(polyhedron.get("vertices", []), dtype=np.float64)
            raw_halfspaces = np.asarray(polyhedron.get("halfspaces", []), dtype=np.float64)
            if raw_vertices.ndim != 2 or raw_vertices.shape[0] < 4 or raw_vertices.shape[1] != 3:
                skipped_polyhedra += 1
                continue
            if raw_halfspaces.ndim != 2 or raw_halfspaces.shape[0] < 4 or raw_halfspaces.shape[1] != 4:
                skipped_polyhedra += 1
                continue

            vertices = deduplicate_vertices(raw_vertices, dedup_eps)
            if vertices.shape[0] < 4:
                skipped_polyhedra += 1
                continue

            faces = triangulate_polyhedron(vertices, raw_halfspaces, plane_eps)
            if not faces:
                skipped_polyhedra += 1
                continue

            base_index = len(all_vertices)
            color = color_from_index(color_cursor, max(total_polyhedra, 1))
            color_cursor += 1

            for vertex in vertices:
                all_vertices.append(vertex)
                all_colors.append(color)
            for face in faces:
                all_faces.append(
                    (base_index + face[0], base_index + face[1], base_index + face[2])
                )

    if not all_vertices or not all_faces:
        raise RuntimeError("Failed to build any corridor mesh faces from the selected trajectories.")

    if skipped_polyhedra > 0:
        print(f"[warning] skipped {skipped_polyhedra} polyhedra because they could not be triangulated cleanly.")

    return (
        np.asarray(all_vertices, dtype=np.float64),
        np.asarray(all_faces, dtype=np.int32),
        np.asarray(all_colors, dtype=np.uint8),
    )


def write_ascii_ply(
    output_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")

        for vertex, color in zip(vertices, colors, strict=True):
            handle.write(
                f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if (args.source_id is None) != (args.target_id is None):
        parser.error("--source-id and --target-id must be used together.")

    args.input = resolve_project_path(args.input)
    if args.output is not None:
        args.output = resolve_project_path(args.output)

    trajectories = load_trajectories(args.input)
    selected = iter_selected_corridors(
        trajectories,
        source_id=args.source_id,
        target_id=args.target_id,
        exact_direction=args.exact_direction,
    )

    output_path = args.output or pick_default_output(args.input, args.source_id, args.target_id)
    vertices, faces, colors = corridor_to_mesh(
        selected_trajectories=selected,
        plane_eps=args.plane_eps,
        dedup_eps=args.dedup_eps,
    )
    write_ascii_ply(output_path, vertices, faces, colors)

    print(
        f"Saved corridor mesh to {output_path} "
        f"({vertices.shape[0]} vertices, {faces.shape[0]} triangles, {len(selected)} trajectories)."
    )


if __name__ == "__main__":
    main()
