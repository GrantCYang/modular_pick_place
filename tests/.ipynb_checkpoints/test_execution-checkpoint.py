# tests/test_execution.py
"""
执行层测试：
  Step 1 — 单阶段冒烟测试：只跑 PRE_GRASP 阶段，确认 TCP 能到达目标位置
  Step 2 — 单个 GraspAction 完整流程：跑完一个物体的 pick-and-place，录制视频
  Step 3 — 完整 Episode：跑完所有物体，统计成功率

用法：
    python tests/test_execution.py --step 1
    python tests/test_execution.py --step 2
    python tests/test_execution.py --step 3 --episodes 10
    python tests/test_execution.py --step 3 --perception vision
    python tests/test_execution.py --step 3 --perception state   ← 默认
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import imageio
import torch
import gymnasium as gym

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                            # noqa
from envs.multi_object_env import MultiObjectPickAndPlaceEnv      # noqa
from perception.state_perception import StatePerception
from perception.vision_perception import VisionPerception
from perception.base import BasePerception, SceneRepresentation
from planning.sequential_planner import SequentialPlanner
from execution.motion_executor import MotionExecutor, ExecutorConfig, Phase

PerceptionMode = Literal["state", "vision"]

# ── 工具 ──────────────────────────────────────────────────────────────────────

def tensor_to_uint8(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return frame


def make_title_frames(
    text: str,
    height: int,
    width: int,
    n_frames: int = 15,
    bg_color: tuple = (20, 20, 20),
    fg_color: tuple = (220, 220, 220),
) -> list[np.ndarray]:
    """
    生成若干帧纯色背景+文字的标题卡，用于分隔不同 episode。
    不依赖 PIL，纯 numpy 实现：用简单的像素块拼出 ASCII 字符。
    若需要更漂亮的文字，可换成 PIL 实现。
    """
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # ── 用 PIL 渲染文字（如果有的话），否则跳过文字只留色块 ──────────────
    try:
        from PIL import Image, ImageDraw, ImageFont
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        # 尝试加载等宽字体，失败则用默认字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 32)
        except Exception:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 32)
            except Exception:
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width  - tw) // 2
        y = (height - th) // 2
        draw.text((x, y), text, fill=fg_color, font=font)
        frame = np.array(img)
    except ImportError:
        # PIL 不可用时，在画面中央画一条亮线作为最低限度的视觉分隔
        cy = height // 2
        frame[cy - 2 : cy + 2, width // 4 : width * 3 // 4] = fg_color

    return [frame.copy() for _ in range(n_frames)]


def make_env(config_path: str, perception_mode: PerceptionMode = "state") -> gym.Env:
    obs_mode = "state" if perception_mode == "state" else "rgb+depth+segmentation"
    return gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode  = "rgb_array",
        obs_mode     = obs_mode,
        control_mode = "pd_ee_delta_pose",
        scene_config = config_path,
    )


def make_perception(env: gym.Env, perception_mode: PerceptionMode) -> BasePerception:
    if perception_mode == "state":
        return StatePerception(env)
    else:
        return VisionPerception(env)


def get_tcp_pos(env) -> np.ndarray:
    p = env.unwrapped.agent.tcp.pose.p
    if isinstance(p, torch.Tensor):
        return p[0].cpu().numpy()
    return np.array(p[0])


def print_scene_summary(scene: SceneRepresentation):
    ta = scene.target_area
    print(f"    感知模式: {scene.perception_mode}  物体数: {scene.n_objects}")
    print(f"    目标区域: center={ta.center.round(3)}  "
          f"size={ta.size.round(3)}  z={ta.table_z:.4f}")
    for obj in scene.objects:
        p = obj.position.round(3)
        print(f"      {obj.object_id:16s} pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})  "
              f"dim={obj.dimensions.round(3)}  conf={obj.confidence:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1：单阶段冒烟测试
# ─────────────────────────────────────────────────────────────────────────────

def test_step1_smoke(config_path: str, perception_mode: PerceptionMode = "state"):
    print("\n" + "═"*55)
    print("  Step 1 — 单阶段冒烟测试")
    print("═"*55)

    env = make_env(config_path, perception_mode="state")
    obs, _ = env.reset(seed=0)

    print(f"  action_space: {env.action_space}")
    print(f"  action_space.shape: {env.action_space.shape}")

    tcp_start = get_tcp_pos(env)
    TARGET    = np.array([0.0, 0.0, 0.20])
    device    = env.unwrapped.agent.tcp.pose.p.device

    print(f"  初始 TCP：{tcp_start.round(3)}")
    print(f"  目标位置：{TARGET}")
    print(f"  device：{device}")

    POS_SCALE  = 0.1
    MOVE_SPEED = 0.06
    THRESHOLD  = 0.008
    frames     = []

    for step in range(200):
        tcp   = get_tcp_pos(env)
        delta = TARGET - tcp
        dist  = np.linalg.norm(delta)
        if dist > MOVE_SPEED:
            delta = delta / dist * MOVE_SPEED

        action_np      = np.zeros(7, dtype=np.float32)
        action_np[0:3] = np.clip(delta / POS_SCALE, -1.0, 1.0)
        action_np[6]   = 1.0
        action_tensor  = torch.tensor(
            action_np[None, :], dtype=torch.float32, device=device
        )

        obs, _, terminated, truncated, _ = env.step(action_tensor)
        frames.append(tensor_to_uint8(env.render()))

        if step % 30 == 0:
            pos = get_tcp_pos(env)
            print(f"  step={step:3d}  TCP={pos.round(3)}  "
                  f"dist={np.linalg.norm(pos - TARGET):.4f}")

        if np.linalg.norm(get_tcp_pos(env) - TARGET) < THRESHOLD:
            print(f"  ✅ 到达目标！step={step}")
            break
        if terminated or truncated:
            break
    else:
        final_dist = np.linalg.norm(get_tcp_pos(env) - TARGET)
        print(f"  {'✅' if final_dist < 0.02 else '❌'} 超时，最终距离={final_dist:.4f}m")

    imageio.mimsave("debug_exec_step1.mp4", frames, fps=20)
    print(f"  视频已保存：debug_exec_step1.mp4")
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2：单个物体的完整 pick-and-place
# ─────────────────────────────────────────────────────────────────────────────

def test_step2_single(
    config_path:      str,
    perception_mode:  PerceptionMode = "state",
):
    print("\n" + "═"*55)
    print(f"  Step 2 — 单个物体完整 pick-and-place  [{perception_mode}]")
    print("═"*55)

    env        = make_env(config_path, perception_mode)
    perception = make_perception(env, perception_mode)
    planner    = SequentialPlanner()
    executor   = MotionExecutor(env)

    obs, _ = env.reset(seed=0)
    scene  = perception.observe(obs)

    print("  感知结果：")
    print_scene_summary(scene)

    seq = planner.plan(scene)

    from planning.base import ActionSequence
    single_seq = ActionSequence(
        actions    = [seq.actions[0]],
        scene_repr = scene,
    )
    executor.load(single_seq)

    target_obj = seq.actions[0].object_id
    print(f"\n  目标物体：{target_obj}")
    print(f"  grasp_pos：{seq.actions[0].grasp_position.round(3)}")
    print(f"  place_pos：{seq.actions[0].place_position.round(3)}")

    frames     = []
    prev_phase = None
    step_count = 0

    while not executor.is_done():
        cur_phase = executor.current_phase
        if cur_phase != prev_phase:
            print(f"\n  ▶ 进入阶段：{cur_phase.name}")
            print(f"    TCP 当前位置：{get_tcp_pos(env).round(3)}")
            prev_phase = cur_phase

        action = executor.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(tensor_to_uint8(env.render()))

        step_count += 1
        if terminated or truncated:
            print("  ⚠️  Episode 提前结束")
            break

    print(f"\n  最终 TCP 位置：{get_tcp_pos(env).round(3)}")
    print(f"  总步数: {step_count}")
    success = env.unwrapped.get_success_info()
    print(f"  目标物体到位：{'✅' if success.get(target_obj, False) else '❌'}")

    out = f"debug_exec_step2_{perception_mode}.mp4"
    imageio.mimsave(out, frames, fps=20)
    print(f"  视频已保存：{out}  ({len(frames)} 帧)")

    if isinstance(perception, VisionPerception):
        perception.reset()
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3：完整 Episode（所有物体）
# ─────────────────────────────────────────────────────────────────────────────

def test_step3_full(
    config_path:     str,
    perception_mode: PerceptionMode = "state",
    n_episodes:      int            = 3,
):
    print("\n" + "═"*55)
    print(f"  Step 3 — 完整 Episode × {n_episodes}  [{perception_mode}]")
    print("═"*55)

    env        = make_env(config_path, perception_mode)
    perception = make_perception(env, perception_mode)
    planner    = SequentialPlanner()
    cfg        = ExecutorConfig()
    executor   = MotionExecutor(env, cfg)

    episode_results = []
    all_frames: list[np.ndarray] = []   # ← 汇总所有 episode 的帧
    frame_h: int | None = None
    frame_w: int | None = None

    for ep in range(n_episodes):
        print(f"\n  ── Episode {ep+1}/{n_episodes} ──")
        obs, _ = env.reset(seed=ep * 13)

        if isinstance(perception, VisionPerception):
            perception.reset()

        scene = perception.observe(obs)

        if ep == 0:
            print("  感知结果：")
            print_scene_summary(scene)

        seq = planner.plan(scene)
        executor.load(seq)

        ep_frames: list[np.ndarray] = []
        prev_phase = None
        step_count = 0
        max_steps  = cfg.max_steps_phase * 9 * seq.n_actions + 50

        while not executor.is_done() and step_count < max_steps:
            cur_phase = executor.current_phase
            if cur_phase != prev_phase:
                obj_idx = executor.current_action_index
                obj_id  = (seq.actions[obj_idx].object_id
                           if obj_idx < seq.n_actions else "—")
                print(f"    [{obj_id}] {cur_phase.name}")
                prev_phase = cur_phase

            action = executor.step()
            obs, reward, terminated, truncated, _ = env.step(action)

            frame = tensor_to_uint8(env.render())
            ep_frames.append(frame)

            # 记录首帧尺寸，后续标题卡用
            if frame_h is None:
                frame_h, frame_w = frame.shape[:2]

            step_count += 1
            if terminated or truncated:
                break

        success_info = env.unwrapped.get_success_info()
        all_ok       = all(success_info.values())
        n_ok         = sum(success_info.values())
        n_total      = len(success_info)

        print(f"\n    成功: {n_ok}/{n_total}  {'✅' if all_ok else '❌'}")
        for obj_id, ok in success_info.items():
            print(f"    {'✅' if ok else '❌'}  {obj_id}")
        print(f"    总步数: {step_count}")
        print(f"  max_steps={max_steps}, n_actions={seq.n_actions}")

        episode_results.append(all_ok)

        # ── 拼接标题卡 + 本 episode 帧 ────────────────────────────────────
        h = frame_h or 480
        w = frame_w or 640
        status_str = "PASS" if all_ok else f"{n_ok}/{n_total}"
        title_text = f"Episode {ep+1}  [{perception_mode}]  {status_str}"
        title_frames = make_title_frames(title_text, h, w, n_frames=15)

        all_frames.extend(title_frames)   # 标题卡（15帧 ≈ 0.75s@20fps）
        all_frames.extend(ep_frames)      # 本 episode 实际内容

    # ── 统计 & 写入总视频 ─────────────────────────────────────────────────
    success_rate = sum(episode_results) / len(episode_results)
    print(f"\n  总成功率：{sum(episode_results)}/{len(episode_results)} "
          f"= {success_rate:.0%}")

    if all_frames:
        out = f"debug_exec_step3_{perception_mode}_all_episodes.mp4"
        imageio.mimsave(out, all_frames, fps=20)
        total_sec = len(all_frames) / 20
        print(f"  视频已保存：{out}  "
              f"({len(all_frames)} 帧 / {total_sec:.1f}s，"
              f"共 {n_episodes} 个 episode)")

    env.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",       type=int,  default=3,
                        help="1=冒烟, 2=单物体, 3=完整episode")
    parser.add_argument("--config",     type=str,  default=None)
    parser.add_argument("--episodes",   type=int,  default=10)
    parser.add_argument("--perception", type=str,  default="state",
                        choices=["state", "vision"],
                        help="感知模式：state（特权状态）或 vision（视觉感知）")
    args = parser.parse_args()

    config = args.config or str(ROOT / "config" / "scene.yaml")
    mode   = args.perception

    if args.step >= 1:
        test_step1_smoke(config, mode)
    if args.step >= 2:
        test_step2_single(config, mode)
    if args.step >= 3:
        test_step3_full(config, mode, args.episodes)