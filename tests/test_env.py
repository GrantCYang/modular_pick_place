# tests/test_env.py
"""
环境冒烟测试：
  - 检查物体数量是否正确（无幽灵 cube）
  - 保存初始帧截图 debug_env_init.png
  - 随机动作 50 步并保存视频 debug_env_random_run.mp4
  - 打印特权状态，确认物体位置合法
  - 工作空间可达性检查：验证所有物体和目标区域4角在 Panda 臂展内

用法：
    python tests/test_env.py
    python tests/test_env.py --config config/scene.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import imageio
import torch
import gymnasium as gym

# ── 确保项目根目录在 sys.path 里 ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                        # noqa: F401
from envs.multi_object_env import MultiObjectPickAndPlaceEnv  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
#  Panda 工作空间几何参数（来自官方 datasheet）
# ─────────────────────────────────────────────────────────────────────────────

# base 位置由 _load_agent 里的 sapien.Pose(p=[-0.615, 0, 0]) 决定
PANDA_BASE_XY        = np.array([-0.615, 0.0])
PANDA_MAX_REACH      = 0.855          # 官方最大臂展，单位 m
# 实际 pick 任务里末端需要一定余量（抓具长度 ~10cm + 姿态冗余）
PANDA_EFFECTIVE_REACH = 0.75          # 保守有效工作半径


# ─────────────────────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_uint8(frame) -> np.ndarray:
    """
    将 ManiSkill 返回的帧（可能是 torch tensor / numpy / 带 batch 维度）
    统一转为 (H, W, 3) uint8 numpy 数组。
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return frame


def check_objects(env, expected_count: int) -> bool:
    """断言场景中物体数量正确（无幽灵 cube）"""
    actual = len(env.unwrapped.objects)
    ok = actual == expected_count
    print(f"{'✅' if ok else '❌'} 物体数量：期望 {expected_count}，实际 {actual}")
    if not ok:
        print("   ⚠️  数量不符！请检查是否仍有父类幽灵 cube 残留。")
    return ok


def print_privileged_state(env) -> dict:
    """打印特权状态，并做简单的合法性检查"""
    state = env.unwrapped.get_privileged_state()
    ta    = state["target"]

    print(f"\n📦 特权状态 — 检测到 {len(state['objects'])} 个物体：")
    for obj in state["objects"]:
        pos  = obj["pose"][:3, 3].round(3)
        xy   = pos[:2]
        x_ok = -0.35 <= xy[0] <= 0.15
        y_ok = -0.20 <= xy[1] <= 0.20
        flag = "✅" if (x_ok and y_ok) else "⚠️ 超出范围"
        print(f"  {flag}  [{obj['id']}] category={obj['category']}, "
              f"pos={pos}, dims={obj['dimensions'].round(3)}")

    print(f"\n🎯 目标区域：center={ta['center']}, "
          f"size={ta['size']}, table_z={ta['table_z']}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
#  工作空间可达性检查
# ─────────────────────────────────────────────────────────────────────────────

def _reach_status(point_xy: np.ndarray) -> Tuple[float, bool, bool]:
    """
    返回 (水平距离, 在最大臂展内, 在有效工作半径内)

    只检查 XY 平面距离：
      - Panda 桌面任务的 Z 高度变化有限（~0–0.3m），水平可达性是主要约束
      - 纵向余量由 PANDA_EFFECTIVE_REACH 的保守值隐式覆盖
    """
    dist  = float(np.linalg.norm(point_xy - PANDA_BASE_XY))
    in_max = dist <= PANDA_MAX_REACH
    in_eff = dist <= PANDA_EFFECTIVE_REACH
    return dist, in_max, in_eff


def _target_area_corners(env) -> List[np.ndarray]:
    """返回目标区域 4 个角的 XY 坐标"""
    ta     = env.unwrapped.scene_cfg.target_area
    cx, cy = float(ta.center[0]), float(ta.center[1])
    hw, hh = float(ta.half[0]),   float(ta.half[1])
    return [
        np.array([cx - hw, cy - hh]),   # 左下
        np.array([cx + hw, cy - hh]),   # 右下
        np.array([cx - hw, cy + hh]),   # 左上
        np.array([cx + hw, cy + hh]),   # 右上
    ]


def check_reachability(env, state: dict) -> bool:
    """
    检查所有物体位置和目标区域 4 角是否在 Panda 工作空间内。

    判据（双层）：
      ❌ 超出最大臂展（0.855m）→ 物理上不可达，硬错误
      ⚠️  在最大臂展内但超出有效半径（0.75m）→ 边缘区域，姿态受限，软警告
      ✅ 在有效工作半径内 → 可达

    返回：True 表示所有点至少在最大臂展内（无硬错误）
    """
    print(f"\n🦾 工作空间可达性检查")
    print(f"   Panda base XY    : {PANDA_BASE_XY}")
    print(f"   最大臂展          : {PANDA_MAX_REACH} m")
    print(f"   有效工作半径（保守）: {PANDA_EFFECTIVE_REACH} m")
    print()

    all_reachable = True
    check_points: List[Tuple[str, np.ndarray]] = []

    # ── 收集物体位置 ───────────────────────────────────────────────────────
    for obj in state["objects"]:
        pos_xy = obj["pose"][:2, 3]                      # 取 T 矩阵前两行的平移
        check_points.append((f"[obj] {obj['id']}", pos_xy))

    # ── 收集目标区域 4 角 ──────────────────────────────────────────────────
    corners = _target_area_corners(env)
    corner_names = ["左下", "右下", "左上", "右上"]
    for name, corner in zip(corner_names, corners):
        check_points.append((f"[target corner {name}]", corner))

    # ── 逐点检查 ───────────────────────────────────────────────────────────
    print(f"   {'点位':<35} {'水平距离':>8}   {'状态'}")
    print(f"   {'-'*35} {'-'*8}   {'-'*12}")

    for label, xy in check_points:
        dist, in_max, in_eff = _reach_status(xy)

        if not in_max:
            symbol = "❌ 超出最大臂展"
            all_reachable = False
        elif not in_eff:
            symbol = "⚠️  边缘区域（姿态受限）"
        else:
            symbol = "✅ 可达"

        print(f"   {label:<35} {dist:>7.3f}m   {symbol}")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print()
    if all_reachable:
        print("   ✅ 所有点位均在最大臂展内，场景配置合法。")
    else:
        print("   ❌ 存在超出最大臂展的点位！请调整 scene.yaml 中的坐标范围。")

    return all_reachable


# ─────────────────────────────────────────────────────────────────────────────
#  主测试流程
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    config_path: str | None = None,
    n_steps:     int  = 50,
    fps:         int  = 20,
    out_image:   str  = "debug_env_init.png",
    out_video:   str  = "debug_env_random_run.mp4",
):
    config_path = config_path or str(ROOT / "config" / "scene.yaml")
    print(f"🚀 测试启动")
    print(f"   配置文件  : {config_path}")
    print(f"   随机步数  : {n_steps}")
    print(f"   输出图片  : {out_image}")
    print(f"   输出视频  : {out_video}")
    print()

    # ── 1. 创建环境 ────────────────────────────────────────────────────────
    env = gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode="rgb_array",
        obs_mode="state",
        scene_config=config_path,
    )
    print(f"✅ 环境创建成功")
    print(f"   obs space : {env.observation_space}")
    print(f"   act space : {env.action_space}")

    # ── 2. Reset ───────────────────────────────────────────────────────────
    obs, info = env.reset(seed=42)
    print(f"\n✅ env.reset() 完成，obs shape = {obs.shape}")

    # ── 3. 物体数量检查 ────────────────────────────────────────────────────
    expected = len(env.unwrapped.scene_cfg.objects)
    check_objects(env, expected)

    # ── 4. 保存初始帧 ──────────────────────────────────────────────────────
    raw_frame  = env.render()
    init_frame = tensor_to_uint8(raw_frame)
    imageio.imwrite(out_image, init_frame)
    print(f"\n📸 初始帧已保存：{out_image}  (shape={init_frame.shape})")

    # ── 5. 打印特权状态 ────────────────────────────────────────────────────
    state = print_privileged_state(env)

    # ── 6. 工作空间可达性检查（新增）─────────────────────────────────────
    reachable = check_reachability(env, state)
    if not reachable:
        print("\n⛔ 可达性检查未通过，建议修正 scene.yaml 后重新测试。")
        # 不强制退出，继续跑完随机步以便获取完整视频信息

    # ── 7. 随机动作 + 录制视频 ─────────────────────────────────────────────
    print(f"\n⚙️  运行 {n_steps} 步随机动作并录制视频...")
    frames      = [init_frame]
    total_reward = 0.0

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += (
            reward.mean().item() if isinstance(reward, torch.Tensor) else float(reward)
        )

        frames.append(tensor_to_uint8(env.render()))

        if terminated or truncated:
            print(f"   Episode 在第 {step + 1} 步结束，重置环境...")
            obs, info = env.reset()

    print(f"   平均奖励：{total_reward / n_steps:.4f}")

    # ── 8. 保存视频 ────────────────────────────────────────────────────────
    imageio.mimsave(out_video, frames, fps=fps)
    print(f"\n🎬 视频已保存：{out_video}  ({len(frames)} 帧, {fps}fps)")

    # ── 9. 成功率统计 ──────────────────────────────────────────────────────
    success_info = env.unwrapped.get_success_info()
    print(f"\n📊 当前 Episode 成功状态：")
    for obj_id, done in success_info.items():
        print(f"   {'✅' if done else '❌'}  {obj_id}")

    env.close()
    print("\n🏁 测试完成！")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiObjectPickAndPlace 环境测试")
    parser.add_argument("--config", type=str,  default=None,
                        help="scene.yaml 路径，默认使用 config/scene.yaml")
    parser.add_argument("--steps",  type=int,  default=50,
                        help="随机动作步数")
    parser.add_argument("--fps",    type=int,  default=20,
                        help="视频帧率")
    parser.add_argument("--image",  type=str,  default="debug_env_init.png")
    parser.add_argument("--video",  type=str,  default="debug_env_random_run.mp4")
    args = parser.parse_args()

    run_test(
        config_path=args.config,
        n_steps=args.steps,
        fps=args.fps,
        out_image=args.image,
        out_video=args.video,
    )