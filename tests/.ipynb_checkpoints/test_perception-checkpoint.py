# tests/test_perception.py
"""
感知层可视化测试（state / vision / compare 三种模式）

用法：
    python tests/test_perception.py                        # 默认 compare 模式
    python tests/test_perception.py --mode state
    python tests/test_perception.py --mode vision
    python tests/test_perception.py --config config/scene.yaml --resets 3

输出文件：
    debug_perception_topdown.png  — 俯视对比图（state 左 / vision 右）
    debug_perception_poses.png    — 3D 位姿坐标轴
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import gymnasium as gym

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                         # noqa
from envs.multi_object_env import MultiObjectPickAndPlaceEnv   # noqa
from perception.state_perception import StatePerception
from perception.vision_perception import VisionPerception
from perception.base import SceneRepresentation

# ── 样式常量 ──────────────────────────────────────────────────────────────────
CATEGORY_STYLE: Dict[str, dict] = {
    "box":      {"color": "#e05c5c", "marker": "s", "prefix": "□"},
    "cylinder": {"color": "#5b9bd5", "marker": "o", "prefix": "○"},
    "mesh":     {"color": "#a0a0a0", "marker": "^", "prefix": "△"},
}
DEFAULT_STYLE = {"color": "#888888", "marker": "x", "prefix": "?"}
SPAWN_X = (-0.15, 0.05)
SPAWN_Y = (-0.15, 0.15)


# ═══════════════════════════════════════════════════════
#  俯视图
# ═══════════════════════════════════════════════════════

def _draw_single_topdown(ax: plt.Axes, scene: SceneRepresentation, title: str):
    """在给定 ax 上画单个感知结果的俯视图。"""
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.35)

    # 生成范围边界
    ax.add_patch(mpatches.Rectangle(
        (SPAWN_X[0], SPAWN_Y[0]),
        SPAWN_X[1] - SPAWN_X[0], SPAWN_Y[1] - SPAWN_Y[0],
        lw=1.2, edgecolor="gray", facecolor="none", linestyle="--",
    ))

    # 目标区域
    ta = scene.target_area
    ax.add_patch(mpatches.FancyBboxPatch(
        (ta.center[0] - ta.size[0] / 2, ta.center[1] - ta.size[1] / 2),
        ta.size[0], ta.size[1],
        boxstyle="round,pad=0.005",
        lw=2, edgecolor="#ccaa00", facecolor="#ffe066", alpha=0.45, zorder=2,
    ))
    ax.text(ta.center[0], ta.center[1], "TARGET",
            ha="center", va="center", fontsize=7,
            color="#886600", fontweight="bold", zorder=3)

    # 每个物体
    for obj in scene.objects:
        style = CATEGORY_STYLE.get(obj.category, DEFAULT_STYLE)
        x, y, z = obj.position
        dx, dy = obj.dimensions[0] / 2, obj.dimensions[1] / 2

        # 包围盒投影
        ax.add_patch(mpatches.Rectangle(
            (x - dx, y - dy), dx * 2, dy * 2,
            lw=1.2, edgecolor=style["color"],
            facecolor=style["color"], alpha=0.22, zorder=3,
        ))

        # 中心点
        ax.scatter(x, y, s=90, c=style["color"],
                   marker=style["marker"], zorder=5,
                   edgecolors="white", linewidths=0.8)

        # 朝向箭头（局部 X 轴）
        local_x = obj.rotation[:, 0]
        alen = max(obj.dimensions[0], 0.02) * 0.7
        ax.annotate("",
            xy=(x + local_x[0] * alen, y + local_x[1] * alen),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=style["color"], lw=1.2),
            zorder=6,
        )

        # 标注
        ax.text(x, y + dy + 0.007,
                f"{style['prefix']}{obj.object_id}\n"
                f"z={z:.3f} conf={obj.confidence:.2f}",
                ha="center", va="bottom", fontsize=6.5,
                color=style["color"], zorder=6)

    # 机器人基座
    ax.plot(-0.615, 0, "k^", markersize=10, zorder=7)
    ax.text(-0.615, 0.015, "Robot\nBase", ha="center", va="bottom",
            fontsize=7, color="black")

    # 原点
    ax.plot(0, 0, "k+", markersize=8, zorder=7)

    # 信息文本框
    lines = [f"n={scene.n_objects}  mode={scene.perception_mode}"]
    for obj in scene.objects:
        p = obj.position.round(3)
        inside = ta.contains(p[:2])
        lines.append(
            f"{'✓' if inside else ' '} {obj.object_id}: "
            f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
        )
    ax.text(0.02, 0.02, "\n".join(lines),
            transform=ax.transAxes, fontsize=6.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            fontfamily="monospace")

    # 自动范围
    all_x = [o.position[0] for o in scene.objects] + [ta.center[0], -0.615, 0]
    all_y = [o.position[1] for o in scene.objects] + [ta.center[1], 0, 0]
    pad = 0.1
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)


def draw_topdown(
    scenes: Dict[str, SceneRepresentation],
    suptitle: str = "Scene Top-Down View",
    out_path: str = "debug_perception_topdown.png",
):
    """
    1 个 scene → 单图；2 个 scene → 左右双栏（同坐标轴）。
    scenes 的 key 会作为子图标题。
    """
    n = len(scenes)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, (label, scene) in zip(axes, scenes.items()):
        _draw_single_topdown(ax, scene, title=f"{label} Perception")

    # 双栏强制统一坐标范围
    if n == 2:
        xlims = [ax.get_xlim() for ax in axes]
        ylims = [ax.get_ylim() for ax in axes]
        xmin, xmax = min(l[0] for l in xlims), max(l[1] for l in xlims)
        ymin, ymax = min(l[0] for l in ylims), max(l[1] for l in ylims)
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📐 俯视图已保存：{out_path}")


# ═══════════════════════════════════════════════════════
#  3D 位姿坐标轴
# ═══════════════════════════════════════════════════════

def draw_pose_axes(
    scene: SceneRepresentation,
    title_prefix: str = "",
    out_path: str = "debug_perception_poses.png",
):
    n = scene.n_objects
    if n == 0:
        print("[draw_pose_axes] 场景为空，跳过。")
        return

    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig  = plt.figure(figsize=(cols * 4, rows * 4))
    fig.suptitle(f"{title_prefix}Object Pose Axes (3D)", fontsize=12)

    for i, obj in enumerate(scene.objects):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"{obj.object_id} ({obj.category})", fontsize=9)

        p = obj.position.astype(np.float64)
        R = obj.rotation.astype(np.float64)
        scale = max(float(obj.dimensions.max()), 0.02) * 0.6

        for j, (color, label) in enumerate(zip(["red", "green", "blue"], ["X", "Y", "Z"])):
            ax.quiver(*p, *(R[:, j] * scale), color=color, lw=2, label=label)

        ax.scatter(*p, color="black", s=50)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.legend(fontsize=7)

        r = max(float(obj.dimensions.max()), 0.05) * 1.5
        ax.set_xlim(p[0] - r, p[0] + r)
        ax.set_ylim(p[1] - r, p[1] + r)
        ax.set_zlim(p[2] - r, p[2] + r)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🧭 位姿坐标轴图已保存：{out_path}")


# ═══════════════════════════════════════════════════════
#  终端报告
# ═══════════════════════════════════════════════════════

def print_scene_report(scene: SceneRepresentation, label: str = ""):
    ta = scene.target_area
    print("\n" + "═" * 58)
    print(f"  [{label}] mode={scene.perception_mode}  "
          f"n={scene.n_objects}  t={scene.timestamp}")
    print("═" * 58)
    print(f"  Target: center={ta.center}  size={ta.size}  z={ta.table_z:.4f}")
    print(f"  Objects:")

    for obj in scene.objects:
        p      = obj.position.round(4)
        det    = np.linalg.det(obj.rotation)
        rot_ok = abs(det - 1.0) < 1e-3
        inside = ta.contains(p[:2])
        flags  = []
        if inside:    flags.append("⚠️ in-target")
        if not rot_ok: flags.append(f"❌ det(R)={det:.4f}")
        flag_str = "  ".join(flags) if flags else "✅"
        print(f"    {flag_str}")
        print(f"      {obj.object_id} | {obj.category} | "
              f"pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) | "
              f"dim={obj.dimensions.round(4)} | "
              f"conf={obj.confidence:.3f}")
    print("═" * 58)


def print_compare_report(
    scene_s: SceneRepresentation,
    scene_v: SceneRepresentation,
):
    """打印 state vs vision 的逐物体位置误差表。"""
    print("\n" + "═" * 65)
    print("  State vs Vision 感知误差")
    print("═" * 65)

    s_map = {o.object_id: o for o in scene_s.objects}
    v_map = {o.object_id: o for o in scene_v.objects}

    only_s = sorted(set(s_map) - set(v_map))
    only_v = sorted(set(v_map) - set(s_map))
    common = sorted(set(s_map) & set(v_map))

    if only_s:
        print(f"  ⚠️  Vision 未检测到: {only_s}")
    if only_v:
        print(f"  ⚠️  State 中没有:    {only_v}")

    print(f"\n  {'ID':<16} {'pos_err(m)':>10} {'dim_err(m)':>10} "
          f"{'conf':>6}  state_pos → vision_pos")
    print("  " + "─" * 62)

    for oid in common:
        os_, ov = s_map[oid], v_map[oid]
        pe = np.linalg.norm(os_.position - ov.position)
        de = np.linalg.norm(os_.dimensions - ov.dimensions)
        sp = os_.position.round(3)
        vp = ov.position.round(3)
        flag = "✅" if pe < 0.02 else ("⚠️ " if pe < 0.05 else "❌")
        print(f"  {flag} {oid:<14} {pe:>10.4f} {de:>10.4f} {ov.confidence:>6.2f}  "
              f"({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) → "
              f"({vp[0]:.3f},{vp[1]:.3f},{vp[2]:.3f})")

    ta_s, ta_v = scene_s.target_area, scene_v.target_area
    print(f"\n  Target Area:  "
          f"center_err={np.linalg.norm(ta_s.center - ta_v.center):.4f}m  "
          f"size_err={np.linalg.norm(ta_s.size - ta_v.size):.4f}m  "
          f"z_err={abs(ta_s.table_z - ta_v.table_z):.4f}m")
    print("═" * 65 + "\n")


# ═══════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════

def run_perception_test(
    config_path: str | None = None,
    mode:        str        = "compare",
    n_resets:    int        = 3,
    out_topdown: str        = "debug_perception_topdown.png",
    out_poses:   str        = "debug_perception_poses.png",
):
    config_path = config_path or str(ROOT / "config" / "scene.yaml")
    mode = mode.lower()
    assert mode in ("state", "vision", "compare"), \
        f"mode 必须是 state/vision/compare，当前: {mode}"

    need_vision = mode in ("vision", "compare")
    # state 模式只需 state obs；vision/compare 需要视觉 obs
    obs_mode = "rgb+depth+segmentation" if need_vision else "state"

    print(f"🔍 感知层测试启动")
    print(f"   模式={mode}  obs_mode={obs_mode}  resets={n_resets}")
    print(f"   config={config_path}\n")

    env = gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode  = "rgb_array",
        obs_mode     = obs_mode,
        scene_config = config_path,
    )

    perc_state  = StatePerception(env)  if mode in ("state",  "compare") else None
    perc_vision = VisionPerception(env) if need_vision                   else None

    for i in range(n_resets):
        print(f"{'─' * 20} Reset #{i+1} {'─' * 20}")
        obs, _ = env.reset(seed=i * 10)

        scene_s = perc_state.observe(obs)  if perc_state  else None
        scene_v = perc_vision.observe(obs) if perc_vision else None

        if scene_s: print_scene_report(scene_s, label="State")
        if scene_v: print_scene_report(scene_v, label="Vision")
        if scene_s and scene_v: print_compare_report(scene_s, scene_v)

        # 最后一次 reset 输出图片
        if i == n_resets - 1:
            tag = f"Reset #{i+1}"

            scenes: Dict[str, SceneRepresentation] = {}
            if scene_s: scenes["State"]  = scene_s
            if scene_v: scenes["Vision"] = scene_v

            draw_topdown(scenes, suptitle=f"Scene Top-Down — {tag}", out_path=out_topdown)
            draw_pose_axes(scene_s or scene_v,
                           title_prefix=f"[{mode.upper()}] ", out_path=out_poses)

        if perc_vision:
            perc_vision.reset()

    env.close()
    print("\n🏁 测试完成！")
    print(f"   {out_topdown}")
    print(f"   {out_poses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument("--mode",    type=str, default="compare",
                        choices=["state", "vision", "compare"])
    parser.add_argument("--resets",  type=int, default=3)
    parser.add_argument("--topdown", type=str, default="debug_perception_topdown.png")
    parser.add_argument("--poses",   type=str, default="debug_perception_poses.png")
    args = parser.parse_args()

    run_perception_test(
        config_path = args.config,
        mode        = args.mode,
        n_resets    = args.resets,
        out_topdown = args.topdown,
        out_poses   = args.poses,
    )