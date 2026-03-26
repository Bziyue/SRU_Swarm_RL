#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from path_utils import resolve_project_path


def parse_region_names(bbox_path: Path) -> List[str]:
    region_names: List[str] = []
    for raw_line in bbox_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("Rectangle:"):
            region_names.append(line.split("Rectangle:", 1)[1].strip())
    if not region_names:
        raise RuntimeError(f"No region names parsed from {bbox_path}")
    return region_names


def parse_txt_paths(txt_path: Path) -> Dict[Tuple[int, int], np.ndarray]:
    lines = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines()]
    index = 0
    paths: Dict[Tuple[int, int], np.ndarray] = {}

    while index < len(lines):
        line = lines[index]
        if not line.startswith("path "):
            index += 1
            continue

        tokens = line.split()
        if len(tokens) < 6:
            raise RuntimeError(f"Malformed path header line: {line}")

        source_id = int(tokens[1])
        target_id = int(tokens[3])
        status = tokens[-1]
        index += 1

        if status == "unreachable":
            continue
        if status != "reachable":
            raise RuntimeError(f"Unexpected path status in line: {line}")

        if index >= len(lines) or not lines[index].startswith("num_points "):
            raise RuntimeError(f"Expected num_points after path header: {line}")

        num_points = int(lines[index].split()[1])
        index += 1
        point_rows = []
        for _ in range(num_points):
            if index >= len(lines):
                raise RuntimeError("Unexpected EOF while reading path points.")
            point_tokens = lines[index].split()
            if len(point_tokens) != 3:
                raise RuntimeError(f"Malformed point row: {lines[index]}")
            point_rows.append([float(point_tokens[0]), float(point_tokens[1]), float(point_tokens[2])])
            index += 1

        if index >= len(lines) or lines[index] != "end_path":
            raise RuntimeError(f"Expected end_path after reading {num_points} points.")
        index += 1

        paths[(source_id, target_id)] = np.asarray(point_rows, dtype=np.float32)

    if not paths:
        raise RuntimeError(f"No reachable paths parsed from {txt_path}")
    return paths


def save_npz(
    output_path: Path,
    region_names: List[str],
    undirected_paths: Dict[Tuple[int, int], np.ndarray],
    save_directed: bool,
) -> None:
    arrays: Dict[str, np.ndarray] = {"region_names": np.asarray(region_names, dtype=object)}

    for (source_id, target_id), path in sorted(undirected_paths.items()):
        arrays[f"path_{source_id}_{target_id}"] = np.asarray(path, dtype=np.float32)
        if save_directed:
            arrays[f"path_{target_id}_{source_id}"] = np.asarray(path[::-1], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert C++ voxel_guidance text path output into the NPZ format expected by visualize_guidance_paths_ui.py"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("region_paths_txt/all_region_pair_paths.txt"),
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
    parser.add_argument(
        "--save-directed",
        action="store_true",
        help="Also write reversed path_<j>_<i> entries. Not needed for the current visualizer, but useful for other tooling.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.input = resolve_project_path(args.input)
    args.bbox_file = resolve_project_path(args.bbox_file)
    args.output = resolve_project_path(args.output)
    region_names = parse_region_names(args.bbox_file)
    undirected_paths = parse_txt_paths(args.input)
    save_npz(args.output, region_names, undirected_paths, save_directed=args.save_directed)
    print(f"Saved {len(undirected_paths)} undirected paths to {args.output}")
    if args.save_directed:
        print(f"Also wrote reversed directed entries for each path.")


if __name__ == "__main__":
    main()
