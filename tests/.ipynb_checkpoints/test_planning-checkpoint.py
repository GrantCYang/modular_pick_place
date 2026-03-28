# tests/test_planning.py
"""
规划层三级测试：
  Level 1 — 纯逻辑：手动构造场景，验证输出的数量/合法性/顺序
  Level 2 — 可视化：matplotlib 俯视图，画出物体位置 + 规划路径 + slot
  Level 3 — 干运行：对接真实环境，不执行动作，只打印规划结果

用法：
    python tests/test_planning.py               # 全部运行
    python tests/test_planning.py --level 1     # 只跑逻辑测试
    python tests/test_planning.py --level 2     # 只跑可视化
    python tests/test_planning.py --level 3     # 只跑 dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import gymnasium as gym

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                            # noqa
from envs.multi_object_env import MultiObjectPickAndPlaceEnv      # noqa
from perception.base import ObjectInfo, TargetArea, SceneRepresentation
from perception.state_perception import StatePerception
from planning.base import ActionSequence
from planning.sequential_planner import SequentialPlanner


# ─────────────────────────────────────────────────────────────────────────────
#  工具：手动构造场景（Level 1/2 不启动仿真器）
# ─────────────────────────────────────────────────────────────────────────────

def _make_pose(x, y, z) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def build_mock_scene() -> SceneRepresentation:
    """构造一个确定性的假场景，用于可重复的逻辑测试"""
    objects = [
        ObjectInfo(
            object_id="box_0",
            category="box",
            pose=_make_pose(-0.13, -0.05, 0.025),
            dimensions=np.array([0.05, 0.05, 0.05]),
        ),
        ObjectInfo(
            object_id="cylinder_0",
            category="cylinder",
            pose=_make_pose(-0.04, -0.13, 0.04),
            dimensions=np.array([0.05, 0.05, 0.08]),
        ),
        ObjectInfo(
            object_id="box_1",
            category="box",
            pose=_make_pose(-0.04, -0.02, 0.03),
            dimensions=np.array([0.04, 0.07, 0.06]),
        ),
    ]
    target_area = TargetArea(
        center=np.array([0.15, 0.0]),
        size=np.array([0.12, 0.12]),
        table_z=0.0,
    )
    return SceneRepresentation(
        objects=objects,
        target_area=target_area,
        perception_mode="mock",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Level 1：纯逻辑测试
# ─────────────────────────────────────────────────────────────────────────────

def test_level1_logic():
    print("\n" + "═"*55)
    print("  Level 1 — 纯逻辑测试（无仿真器）")
    print("═"*55)

    scene = build_mock_scene()
    planner = SequentialPlanner()
    seq = planner.plan(scene)
    ta = scene.target_area

    all_pass = True

    # ① 动作数量 == 物体数量
    n_ok = seq.n_actions == scene.n_objects
    print(f"\n{'✅' if n_ok else '❌'} 动作数量：{seq.n_actions} == {scene.n_objects}")
    all_pass &= n_ok

    # ② 每个放置点在目标区域内
    print("\n  放置点合法性（应全部在目标区域内）：")
    for act in seq.actions:
        pp = act.place_position
        inside = ta.contains(pp[:2])
        print(f"  {'✅' if inside else '❌'}  {act.object_id}  →  place=({pp[0]:.3f}, {pp[1]:.3f}, {pp[2]:.3f})")
        all_pass &= inside

    # ③ 抓取点 Z 合理（应 > 0，< 0.3）
    print("\n  抓取点 Z 合法性（应在桌面以上，不超过 30cm）：")
    for act in seq.actions:
        gp = act.grasp_position
        z_ok = 0.0 < gp[2] < 0.3
        print(f"  {'✅' if z_ok else '❌'}  {act.object_id}  →  grasp_z={gp[2]:.3f}")
        all_pass &= z_ok

    # ④ 旋转矩阵合法（det ≈ 1）
    print("\n  位姿旋转矩阵合法性（det ≈ 1）：")
    for act in seq.actions:
        for name, pose in [("grasp", act.grasp_pose), ("place", act.place_pose)]:
            det = np.linalg.det(pose[:3, :3])
            rot_ok = abs(det - 1.0) < 1e-4
            print(f"  {'✅' if rot_ok else '❌'}  {act.object_id} {name}: det(R)={det:.6f}")
            all_pass &= rot_ok

    # ⑤ 抓取顺序：第一个抓的物体应该离目标最远
    print("\n  抓取顺序（从远到近）：")
    ta_c = np.array([ta.center[0], ta.center[1], 0.0])
    for i, act in enumerate(seq.actions):
        obj = scene.get_object_by_id(act.object_id)
        dist = np.linalg.norm(obj.position - ta_c)
        print(f"  [{i+1}] {act.object_id}  距目标中心 {dist:.3f}m")

    dists = []
    for act in seq.actions:
        obj = scene.get_object_by_id(act.object_id)
        dists.append(np.linalg.norm(obj.position - ta_c))
    order_ok = all(dists[i] >= dists[i+1] for i in range(len(dists)-1))
    print(f"  {'✅' if order_ok else '❌'}  顺序正确（远→近）")
    all_pass &= order_ok

    # ⑥ slot 不重叠：任意两个放置点的距离 > 最小物体尺寸
    print("\n  放置点互不重叠：")
    place_positions = [a.place_position for a in seq.actions]
    overlap_ok = True
    for i in range(len(place_positions)):
        for j in range(i+1, len(place_positions)):
            dist = np.linalg.norm(place_positions[i][:2] - place_positions[j][:2])
            min_safe = 0.04   # 最小 4cm 间距
            ok = dist > min_safe
            if not ok:
                print(f"  ❌  slot_{i} ↔ slot_{j}  距离={dist:.3f}m  < {min_safe}m（太近！）")
                overlap_ok = False
    if overlap_ok:
        print("  ✅  所有放置点间距合格")
    all_pass &= overlap_ok

    print("\n" + "─"*55)
    print(f"  {'✅ 全部通过' if all_pass else '❌ 存在失败项，请检查上方输出'}")
    print("═"*55)
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  Level 2：可视化测试
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "box":      "#e05c5c",
    "cylinder": "#5b9bd5",
    "box_1":    "#5be08a",
    "slot":     "#ffe066",
    "path":     "#aaaaaa",
    "arrow":    "#444444",
}

def _obj_color(obj: ObjectInfo, idx: int) -> str:
    palette = ["#e05c5c", "#5b9bd5", "#5be08a", "#e0a030", "#9b5be0"]
    return palette[idx % len(palette)]


def draw_plan(
    scene: SceneRepresentation,
    seq: ActionSequence,
    out_path: str = "debug_planning_topdown.png",
):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_title("Planning Result — Top-Down View (XY Plane)", fontsize=13, pad=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle="--", alpha=0.35)

    ta = scene.target_area

    # ── 目标区域 ──────────────────────────────────────────────────────────
    target_rect = mpatches.FancyBboxPatch(
        (ta.center[0] - ta.size[0]/2, ta.center[1] - ta.size[1]/2),
        ta.size[0], ta.size[1],
        boxstyle="round,pad=0.003",
        linewidth=2, edgecolor="#ccaa00",
        facecolor="#ffe066", alpha=0.4, zorder=2,
    )
    ax.add_patch(target_rect)
    ax.text(ta.center[0], ta.center[1], "TARGET",
            ha="center", va="center", fontsize=8,
            color="#886600", fontweight="bold", zorder=3)

    # ── 生成范围边界 ──────────────────────────────────────────────────────
    spawn_rect = mpatches.Rectangle(
        (-0.15, -0.15), 0.20, 0.30,
        linewidth=1.2, edgecolor="gray",
        facecolor="none", linestyle="--", zorder=1,
    )
    ax.add_patch(spawn_rect)

    # ── 物体和规划路径 ────────────────────────────────────────────────────
    for rank, act in enumerate(seq.actions):
        obj = scene.get_object_by_id(act.object_id)
        color = _obj_color(obj, scene.objects.index(obj))

        ox, oy = obj.position[0], obj.position[1]
        gx, gy = act.grasp_position[0], act.grasp_position[1]
        px, py = act.place_position[0], act.place_position[1]
        dx, dy = obj.dimensions[0]/2, obj.dimensions[1]/2

        # 物体当前位置（包围盒）
        bbox = mpatches.Rectangle(
            (ox-dx, oy-dy), dx*2, dy*2,
            linewidth=2, edgecolor=color,
            facecolor=color, alpha=0.3, zorder=3,
        )
        ax.add_patch(bbox)
        ax.scatter(ox, oy, s=100, c=color, zorder=5, edgecolors="white", linewidths=1)
        ax.text(ox, oy + dy + 0.006,
                f"[{rank+1}] {act.object_id}",
                ha="center", va="bottom", fontsize=8,
                color=color, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        # slot（目标位置）
        sdx, sdy = obj.dimensions[0]/2, obj.dimensions[1]/2
        slot_rect = mpatches.Rectangle(
            (px-sdx, py-sdy), sdx*2, sdy*2,
            linewidth=1.5, edgecolor=color,
            facecolor=color, alpha=0.15,
            linestyle="--", zorder=3,
        )
        ax.add_patch(slot_rect)
        ax.scatter(px, py, s=60, c=color, marker="x", zorder=5, linewidths=2)
        ax.text(px, py - sdy - 0.006,
                f"slot_{rank+1}",
                ha="center", va="top", fontsize=7,
                color=color, zorder=6)

        # 路径箭头（物体 → slot）
        ax.annotate(
            "",
            xy=(px, py), xytext=(ox, oy),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.5,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=4,
        )

        # 顺序数字标在箭头中点
        mid_x = (ox + px) / 2
        mid_y = (oy + py) / 2
        ax.text(mid_x, mid_y,
                f"#{rank+1}",
                ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.7, edgecolor=color),
                zorder=7)

    # ── 机器人基座 ────────────────────────────────────────────────────────
    ax.plot(-0.615, 0, "k^", markersize=12, zorder=8)
    ax.text(-0.615, 0.015, "Robot", ha="center", fontsize=8, color="black")

    # ── 坐标原点 ──────────────────────────────────────────────────────────
    ax.plot(0, 0, "k+", markersize=10, zorder=8)

    # ── 信息框 ────────────────────────────────────────────────────────────
    planner = SequentialPlanner()
    info_text = planner.describe_plan(seq)
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=7, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", alpha=0.85))

    # ── 轴范围 & 保存 ─────────────────────────────────────────────────────
    all_x = [o.position[0] for o in scene.objects] + [ta.center[0], -0.615]
    all_y = [o.position[1] for o in scene.objects] + [ta.center[1], 0]
    for act in seq.actions:
        all_x.append(act.place_position[0])
        all_y.append(act.place_position[1])
    pad = 0.10
    ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
    ax.set_ylim(min(all_y)-pad, max(all_y)+pad)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📐 规划俯视图已保存：{out_path}")


def test_level2_visual(out_path: str = "debug_planning_topdown.png"):
    print("\n" + "═"*55)
    print("  Level 2 — 可视化测试")
    print("═"*55)
    scene = build_mock_scene()
    planner = SequentialPlanner()
    seq = planner.plan(scene)
    draw_plan(scene, seq, out_path)
    print(planner.describe_plan(seq))


# ─────────────────────────────────────────────────────────────────────────────
#  Level 3：干运行（对接真实环境，不执行物理动作）
# ─────────────────────────────────────────────────────────────────────────────

def test_level3_dryrun(config_path: str | None = None):
    print("\n" + "═"*55)
    print("  Level 3 — 干运行（真实环境 + 感知，不执行动作）")
    print("═"*55)

    config_path = config_path or str(ROOT / "config" / "scene.yaml")

    env = gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode="rgb_array",
        obs_mode="state",
        scene_config=config_path,
    )
    perception = StatePerception(env)
    planner    = SequentialPlanner()

    n_trials = 3
    all_plans_valid = True

    for trial in range(n_trials):
        print(f"\n  ── Trial {trial+1}/{n_trials} ──")
        obs, _ = env.reset(seed=trial * 7)

        scene = perception.observe(obs)
        seq   = planner.plan(scene)
        ta    = scene.target_area

        print(planner.describe_plan(seq))

        # 验证每个放置点在目标区域内
        trial_ok = True
        for act in seq.actions:
            pp = act.place_position
            inside = ta.contains(pp[:2])
            if not inside:
                print(f"  ❌  {act.object_id} 的 place_pose 不在目标区域内！"
                      f"  place_xy=({pp[0]:.3f}, {pp[1]:.3f})")
                trial_ok = False
        if trial_ok:
            print(f"  ✅  Trial {trial+1} 规划合法")
        all_plans_valid &= trial_ok

    env.close()

    print("\n" + "─"*55)
    print(f"  {'✅ 全部 Trial 规划合法' if all_plans_valid else '❌ 存在非法规划'}")
    print("═"*55)
    return all_plans_valid


# ─────────────────────────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", type=int, default=0,
        help="0=全部, 1=逻辑, 2=可视化, 3=干运行"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out",    type=str, default="debug_planning_topdown.png")
    args = parser.parse_args()

    results = {}
    if args.level in (0, 1):
        results["level1"] = test_level1_logic()
    if args.level in (0, 2):
        test_level2_visual(args.out)
        results["level2"] = True   # 可视化不返回 pass/fail
    if args.level in (0, 3):
        results["level3"] = test_level3_dryrun(args.config)

    print("\n🏁 测试完成")
    for k, v in results.items():
        print(f"   {k}: {'✅ PASS' if v else '❌ FAIL'}")