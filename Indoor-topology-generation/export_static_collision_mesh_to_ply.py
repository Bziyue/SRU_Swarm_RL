#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom

from path_utils import resolve_project_path


def transform_points(points: list[Gf.Vec3f], transform: Gf.Matrix4d) -> np.ndarray:
    transformed: list[list[float]] = []
    for point in points:
        value = transform.Transform(Gf.Vec3d(point[0], point[1], point[2]))
        transformed.append([float(value[0]), float(value[1]), float(value[2])])
    return np.asarray(transformed, dtype=np.float32)


def triangulate_faces(face_vertex_counts, face_vertex_indices) -> np.ndarray:
    faces: list[list[int]] = []
    index = 0
    for count in face_vertex_counts:
        count_int = int(count)
        for offset in range(count_int - 2):
            faces.append(
                [
                    int(face_vertex_indices[index]),
                    int(face_vertex_indices[index + 1 + offset]),
                    int(face_vertex_indices[index + 2 + offset]),
                ]
            )
        index += count_int
    return np.asarray(faces, dtype=np.int32)


def collect_stage_meshes(stage: Usd.Stage) -> trimesh.Trimesh:
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_vertex_counts or not face_vertex_indices:
            continue

        transform = xform_cache.GetLocalToWorldTransform(prim)
        vertices_np = transform_points(list(points), transform)
        faces_np = triangulate_faces(face_vertex_counts, face_vertex_indices)
        if len(vertices_np) == 0 or len(faces_np) == 0:
            continue

        all_vertices.append(vertices_np)
        all_faces.append(faces_np + vertex_offset)
        vertex_offset += len(vertices_np)

    if not all_vertices or not all_faces:
        raise RuntimeError("No mesh geometry found in stage.")

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the static training mesh to a PLY for Open3D visualization.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("DR_static_mesh.usdc"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DR_static_mesh_collision.ply"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.input = resolve_project_path(args.input)
    args.output = resolve_project_path(args.output)
    stage = Usd.Stage.Open(str(args.input))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {args.input}")

    mesh = collect_stage_meshes(stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.output))
    print(f"Exported {len(mesh.vertices)} vertices and {len(mesh.faces)} faces to {args.output}")


if __name__ == "__main__":
    main()
