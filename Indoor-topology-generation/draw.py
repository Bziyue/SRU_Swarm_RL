from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

from path_utils import resolve_project_path


def parse_and_plot_rectangles(filepath: Path) -> None:
    rectangles = {}
    current_rect = None

    # 1. 解析 txt 文件内容
    try:
        with filepath.open('r', encoding='utf-8') as file:
            for line in file:
                # 匹配矩形名称
                if line.startswith("Rectangle:"):
                    current_rect = line.split("Rectangle:")[1].strip()
                    rectangles[current_rect] = []
                # 匹配坐标行
                elif "Corner" in line and current_rect:
                    # 使用正则提取 X, Y, Z 的数值
                    match = re.search(r"X:\s*([-\d.]+),\s*Y:\s*([-\d.]+),\s*Z:\s*([-\d.]+)", line)
                    if match:
                        x, y, z = map(float, match.groups())
                        rectangles[current_rect].append([x, y, z])
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}。请检查路径。")
        return

    # 检查是否成功读取数据
    if not rectangles:
        print("未能在文件中解析到矩形数据，请检查 txt 文件的格式。")
        return

    # 2. 设置 2D 绘图环境，按 XY 平面俯视展示
    fig, ax = plt.subplots(figsize=(10, 8))

    # 3. 处理并绘制每个矩形
    for name, corners in rectangles.items():
        if len(corners) < 3:
            continue  # 跳过无效的形状
            
        pts = np.array(corners)
        
        # 仅在 XY 平面排序和绘制，忽略 Z 轴高度
        pts_xy = pts[:, :2]
        centroid = np.mean(pts_xy, axis=0)
        
        # 使用相对中心的角度进行排序，确保按照顺时针或逆时针连接
        angles = np.arctan2(pts_xy[:, 1] - centroid[1], pts_xy[:, 0] - centroid[0])
        sort_order = np.argsort(angles)
        sorted_pts = pts_xy[sort_order]
        
        # 复制第一个点到数组末尾，以闭合矩形线框
        closed_pts = np.vstack((sorted_pts, sorted_pts[0]))

        # 在平面图中绘制轮廓和角点
        ax.plot(closed_pts[:, 0], closed_pts[:, 1], color='blue', alpha=0.7)
        ax.scatter(pts_xy[:, 0], pts_xy[:, 1], color='red', s=15)

    # 设置图表属性
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_title('Top-Down XY Visualization of Exported Rectangles')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)

    print(f"解析完成，共绘制 {len(rectangles)} 个矩形。")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot region rectangles from the bbox txt file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("DR_Surface_BBox_Data.txt"),
        help="BBox txt file inside Indoor-topology-generation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parse_and_plot_rectangles(resolve_project_path(args.input))


if __name__ == "__main__":
    main()
