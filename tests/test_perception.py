# tests/test_perception.py
"""
感知层可视化测试：
  - 用 StatePerception 读取场景
  - 用 matplotlib 画出俯视图（XY 平面）
  - 对比 debug_env_init.png 验证位置正确性

用法：
    python tests/test_perception.py
    python tests/test_perception.py --config config/scene.yaml --steps 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 服务器无头环境，不弹窗
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import gymnasium as gym
import torch

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                        # noqa
from envs.multi_object_env import MultiObjectPickAndPlaceEnv  # noqa
from perception.state_perception import StatePerception
from perception.base import SceneRepresentation, ObjectInfo


# ── 绘图工具 ──────────────────────────────────────────────────────────────────

# 每种 category 对应的颜色和形状（和 YAML 里的视觉颜色尽量对应）
CATEGORY_STYLE = {
    "box":      {"color": "#e05c5c", "marker": "s", "label_prefix": "□"},
    "cylinder": {"color": "#5b9bd5", "marker": "o", "label_prefix": "○"},
    "mesh":     {"color": "#a0a0a0", "marker": "^", "label_prefix": "△"},
}
DEFAULT_STYLE = {"color": "#888888", "marker": "x", "label_prefix": "?"}


def draw_scene_topdown(
    scene: SceneRepresentation,
    title: str = "Scene Top-Down View (XY Plane)",
    out_path: str = "debug_perception_topdown.png",
    spawn_x: tuple = (-0.15, 0.05),
    spawn_y: tuple = (-0.15, 0.15),
):
    """
    绘制场景俯视图，包含：
      - 目标区域（黄色矩形）
      - 物体位置（带 ID 标注）
      - 物体朝向（用箭头表示 X 轴方向）
      - 物体包围盒投影（XY 平面矩形）
      - 生成范围边界（灰色虚线）
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── 生成范围边界 ──────────────────────────────────────────────────────
    spawn_rect = mpatches.Rectangle(
        (spawn_x[0], spawn_y[0]),
        spawn_x[1] - spawn_x[0],
        spawn_y[1] - spawn_y[0],
        linewidth=1.5,
        edgecolor="gray",
        facecolor="none",
        linestyle="--",
        label="Spawn Range",
    )
    ax.add_patch(spawn_rect)

    # ── 目标区域 ──────────────────────────────────────────────────────────
    ta = scene.target_area
    target_rect = mpatches.FancyBboxPatch(
        (ta.center[0] - ta.size[0] / 2, ta.center[1] - ta.size[1] / 2),
        ta.size[0], ta.size[1],
        boxstyle="round,pad=0.005",
        linewidth=2,
        edgecolor="#ccaa00",
        facecolor="#ffe066",
        alpha=0.45,
        label=f"Target Area\n({ta.center[0]:.2f}, {ta.center[1]:.2f})",
        zorder=2,
    )
    ax.add_patch(target_rect)
    ax.text(
        ta.center[0], ta.center[1], "TARGET",
        ha="center", va="center", fontsize=8,
        color="#886600", fontweight="bold", zorder=3,
    )

    # ── 每个物体 ──────────────────────────────────────────────────────────
    legend_handles = []
    for obj in scene.objects:
        style = CATEGORY_STYLE.get(obj.category, DEFAULT_STYLE)
        x, y, z = obj.position

        # 包围盒投影（XY 平面矩形）
        dx, dy = obj.dimensions[0] / 2, obj.dimensions[1] / 2
        bbox_rect = mpatches.Rectangle(
            (x - dx, y - dy), dx * 2, dy * 2,
            linewidth=1.5,
            edgecolor=style["color"],
            facecolor=style["color"],
            alpha=0.25,
            zorder=3,
        )
        ax.add_patch(bbox_rect)

        # 物体中心点
        sc = ax.scatter(
            x, y,
            s=120,
            c=style["color"],
            marker=style["marker"],
            zorder=5,
            edgecolors="white",
            linewidths=1,
        )

        # 朝向箭头（旋转矩阵第一列 = 局部 X 轴）
        rot = obj.rotation
        local_x = rot[:, 0]                           # world 坐标系下的 X 轴方向
        arrow_len = max(obj.dimensions[0], 0.02) * 0.8
        ax.annotate(
            "",
            xy=(x + local_x[0] * arrow_len, y + local_x[1] * arrow_len),
            xytext=(x, y),
            arrowprops=dict(
                arrowstyle="->",
                color=style["color"],
                lw=1.5,
            ),
            zorder=6,
        )

        # ID 标注
        ax.text(
            x, y + dy + 0.008,
            f"{style['label_prefix']} {obj.object_id}\n"
            f"z={z:.3f}  conf={obj.confidence:.2f}",
            ha="center", va="bottom", fontsize=7.5,
            color=style["color"], zorder=6,
        )

        # 图例条目（每种 category 只加一次）
        already = [h.get_label() for h in legend_handles]
        if obj.category not in already:
            legend_handles.append(
                mpatches.Patch(color=style["color"], label=obj.category)
            )

    # ── 机器人基座位置（固定在 x=-0.615） ────────────────────────────────
    ax.plot(-0.615, 0, "k^", markersize=12, label="Robot Base", zorder=7)
    ax.text(-0.615, 0.02, "Robot\nBase",
            ha="center", va="bottom", fontsize=8, color="black")

    # ── 坐标原点 ──────────────────────────────────────────────────────────
    ax.plot(0, 0, "k+", markersize=10, zorder=7)
    ax.text(0.005, 0.005, "Origin", fontsize=7, color="black")

    # ── 统计信息文本框 ────────────────────────────────────────────────────
    stats_lines = [
        f"Objects: {scene.n_objects}",
        f"Mode: {scene.perception_mode}",
        f"Timestamp: {scene.timestamp}",
        "",
    ]
    for obj in scene.objects:
        x, y, z = obj.position.round(3)
        in_target = ta.contains(obj.position[:2])
        stats_lines.append(
            f"{'✓' if in_target else '✗'} {obj.object_id}: "
            f"({x}, {y}, {z})  {'← IN TARGET' if in_target else ''}"
        )

    stats_text = "\n".join(stats_lines)
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        fontfamily="monospace",
    )

    # ── 图例 & 保存 ───────────────────────────────────────────────────────
    legend_handles += [
        mpatches.Patch(color="#ffe066", label="Target Area",
                       linewidth=2, edgecolor="#ccaa00"),
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   label="Spawn Range"),
        plt.Line2D([0], [0], color="black", marker="^",
                   linestyle="none", label="Robot Base"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # 自动调整坐标轴范围，留一点 padding
    all_x = [o.position[0] for o in scene.objects] + [ta.center[0], -0.615, 0]
    all_y = [o.position[1] for o in scene.objects] + [ta.center[1], 0, 0]
    pad = 0.08
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📐 俯视图已保存：{out_path}")


def draw_pose_axes(
    scene: SceneRepresentation,
    out_path: str = "debug_perception_poses.png",
):
    """
    3D 坐标轴图：验证每个物体的旋转矩阵是否正确。
    每个物体单独一个子图，画出 X/Y/Z 三轴。
    """
    n = scene.n_objects
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 4, rows * 4))
    fig.suptitle("Object Pose Axes (3D)", fontsize=13)

    for i, obj in enumerate(scene.objects):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"{obj.object_id}\n({obj.category})", fontsize=9)

        p = obj.position
        R = obj.rotation
        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]
        scale = max(obj.dimensions) * 0.6

        for j in range(3):
            ax.quiver(
                p[0], p[1], p[2],
                R[0, j] * scale, R[1, j] * scale, R[2, j] * scale,
                color=colors[j], linewidth=2, label=labels[j],
            )

        ax.scatter(*p, color="black", s=50, zorder=5)

        # 轴标签
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.legend(fontsize=7, loc="upper left")

        # 设置合理的范围
        r = max(obj.dimensions) * 1.5
        ax.set_xlim(p[0] - r, p[0] + r)
        ax.set_ylim(p[1] - r, p[1] + r)
        ax.set_zlim(p[2] - r, p[2] + r)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🧭 位姿坐标轴图已保存：{out_path}")


def print_scene_report(scene: SceneRepresentation):
    """在终端打印结构化的场景报告"""
    ta = scene.target_area
    print("\n" + "═" * 55)
    print(f"  SceneRepresentation Report")
    print(f"  mode={scene.perception_mode}  |  n_objects={scene.n_objects}")
    print("═" * 55)

    print(f"\n  目标区域:")
    print(f"    center   = {ta.center}")
    print(f"    size     = {ta.size}")
    print(f"    table_z  = {ta.table_z}")

    # 验证目标区域采样位置
    placements = ta.sample_placement_positions(scene.n_objects)
    print(f"\n  目标区域内的规划放置点（{scene.n_objects} 个）:")
    for i, pos in enumerate(placements):
        inside = ta.contains(pos[:2])
        flag = "✅" if inside else "❌"
        print(f"    {flag}  slot_{i}: {pos.round(4)}")

    print(f"\n  物体列表:")
    all_ok = True
    for obj in scene.objects:
        pos = obj.position.round(4)
        in_target = ta.contains(pos[:2])
        # 位姿矩阵合法性：旋转矩阵行列式应为 1
        det = np.linalg.det(obj.rotation)
        rot_ok = abs(det - 1.0) < 1e-4
        flag = "✅" if (not in_target and rot_ok) else ""
        if in_target:
            flag = "⚠️  已在目标区域"
        if not rot_ok:
            flag += f"  ❌ 旋转矩阵行列式={det:.4f}"
            all_ok = False

        print(f"    {flag}")
        print(f"      id         = {obj.object_id}")
        print(f"      category   = {obj.category}")
        print(f"      position   = {pos}")
        print(f"      dimensions = {obj.dimensions.round(4)}")
        print(f"      det(R)     = {det:.6f}  {'✅' if rot_ok else '❌'}")
        print(f"      is_grasped = {obj.is_grasped}")
        print(f"      confidence = {obj.confidence}")

    print("\n" + "─" * 55)
    if all_ok:
        print("  ✅ 所有检查通过")
    else:
        print("  ❌ 存在异常，请检查上方 ❌ 项目")
    print("═" * 55 + "\n")


# ── 主测试流程 ────────────────────────────────────────────────────────────────

def run_perception_test(
    config_path: str | None = None,
    out_topdown: str = "debug_perception_topdown.png",
    out_poses: str = "debug_perception_poses.png",
    n_resets: int = 3,                # 多次 reset，验证随机化的合理性
):
    config_path = config_path or str(ROOT / "config" / "scene.yaml")
    print(f"🔍 感知层测试启动")
    print(f"   配置文件  : {config_path}")
    print(f"   重置次数  : {n_resets}（验证随机化稳定性）\n")

    env = gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode="rgb_array",
        obs_mode="state",
        scene_config=config_path,
    )
    perception = StatePerception(env)

    for i in range(n_resets):
        print(f"{'─'*20} Reset #{i+1} {'─'*20}")
        obs, _ = env.reset(seed=i * 10)
        scene = perception.observe(obs)

        # 终端报告
        print_scene_report(scene)

        # 只对最后一次 reset 保存图片（避免覆盖太多次）
        if i == n_resets - 1:
            draw_scene_topdown(
                scene,
                title=f"Scene Top-Down View — Reset #{i+1}",
                out_path=out_topdown,
            )
            draw_pose_axes(scene, out_path=out_poses)

    env.close()
    print("🏁 感知层测试完成！")
    print(f"   俯视图 → {out_topdown}")
    print(f"   位姿图 → {out_poses}")
    print(f"\n   ⬇️  请下载这两张图片，与 debug_env_init.png 对比验证位置一致性")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="感知层可视化测试")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resets", type=int, default=3, help="随机重置次数")
    parser.add_argument("--topdown", type=str, default="debug_perception_topdown.png")
    parser.add_argument("--poses",   type=str, default="debug_perception_poses.png")
    args = parser.parse_args()

    run_perception_test(
        config_path=args.config,
        out_topdown=args.topdown,
        out_poses=args.poses,
        n_resets=args.resets,
    )