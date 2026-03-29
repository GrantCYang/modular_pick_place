# demo.py
"""
Demo script — 完整 Pick-and-Place Pipeline

用法：
    python demo.py                            # state 感知，10 轮
    python demo.py --perception vision        # 视觉感知，10 轮
    python demo.py --perception state --episodes 5
    python demo.py --config path/to/scene.yaml --perception vision --episodes 10

输出：
    - 控制台：每轮每物体成功/失败，最终汇总成功率
    - 视频：demo_state_all_episodes.mp4 / demo_vision_all_episodes.mp4
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

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                            # noqa
from envs.multi_object_env import MultiObjectPickAndPlaceEnv      # noqa
from perception.state_perception import StatePerception
from perception.vision_perception import VisionPerception
from perception.base import BasePerception, SceneRepresentation
from planning.sequential_planner import SequentialPlanner
from execution.motion_executor import MotionExecutor, ExecutorConfig

PerceptionMode = Literal["state", "vision"]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

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
    """生成标题卡帧，用于分隔不同 episode。"""
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 32
            )
        except Exception:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 32)
            except Exception:
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - tw) // 2, (height - th) // 2), text,
                  fill=fg_color, font=font)
        frame = np.array(img)
    except ImportError:
        cy = height // 2
        frame[cy - 2 : cy + 2, width // 4 : width * 3 // 4] = fg_color
    return [frame.copy() for _ in range(n_frames)]


def print_scene_summary(scene: SceneRepresentation):
    ta = scene.target_area
    print(f"    感知模式: {scene.perception_mode}  物体数: {scene.n_objects}")
    print(f"    目标区域: center={ta.center.round(3)}  "
          f"size={ta.size.round(3)}  z={ta.table_z:.4f}")
    for obj in scene.objects:
        p = obj.position.round(3)
        print(f"      {obj.object_id:16s} "
              f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})  "
              f"dim={obj.dimensions.round(3)}  conf={obj.confidence:.2f}")


def make_env(config_path: str, perception_mode: PerceptionMode) -> gym.Env:
    obs_mode = "state" if perception_mode == "state" else "rgb+depth+segmentation"
    return gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode  = "rgb_array",
        obs_mode     = obs_mode,
        control_mode = "pd_ee_delta_pose",
        scene_config = config_path,
    )


def make_perception(env: gym.Env, mode: PerceptionMode) -> BasePerception:
    return StatePerception(env) if mode == "state" else VisionPerception(env)


# ── 主 Pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(
    config_path:     str,
    perception_mode: PerceptionMode = "state",
    n_episodes:      int            = 10,
):
    print("\n" + "═" * 60)
    print(f"  Modular Pick-and-Place Demo")
    print(f"  Perception: {perception_mode.upper()}   Episodes: {n_episodes}")
    print("═" * 60)

    env        = make_env(config_path, perception_mode)
    perception = make_perception(env, perception_mode)
    planner    = SequentialPlanner()
    cfg        = ExecutorConfig()
    executor   = MotionExecutor(env, cfg)

    episode_results: list[bool]    = []
    all_frames:      list[np.ndarray] = []
    frame_h: int | None = None
    frame_w: int | None = None

    for ep in range(n_episodes):
        print(f"\n  ── Episode {ep + 1}/{n_episodes} ──")
        obs, _ = env.reset(seed=ep * 13)

        if isinstance(perception, VisionPerception):
            perception.reset()

        # ── 感知 ──────────────────────────────────────────────────────────
        scene = perception.observe(obs)
        if ep == 0:
            print("  初始场景感知：")
            print_scene_summary(scene)

        # ── 规划 ──────────────────────────────────────────────────────────
        seq = planner.plan(scene)
        executor.load(seq)

        # ── 执行 ──────────────────────────────────────────────────────────
        ep_frames:  list[np.ndarray] = []
        prev_phase  = None
        step_count  = 0
        # 安全步数上限：每阶段最大步 × 阶段数(9) × 物体数 + 冗余
        max_steps   = cfg.max_steps_phase * 9 * seq.n_actions + 100

        while step_count < max_steps:
            # ★ 关键修复：只有 executor 真正完成才退出，
            #   不依赖环境的 terminated/truncated，
            #   避免最后一个物体尚未放下就结算。
            if executor.is_done():
                break

            cur_phase = executor.current_phase
            if cur_phase != prev_phase:
                obj_idx = executor.current_action_index
                obj_id  = (seq.actions[obj_idx].object_id
                           if obj_idx < seq.n_actions else "—")
                print(f"    [{obj_id}] ▶ {cur_phase.name}")
                prev_phase = cur_phase

            action = executor.step()
            obs, _reward, terminated, truncated, _ = env.step(action)

            frame = tensor_to_uint8(env.render())
            ep_frames.append(frame)
            if frame_h is None:
                frame_h, frame_w = frame.shape[:2]

            step_count += 1

            # 环境强制结束（超时/碰撞）时才真正中断
            if truncated:
                print("    ⚠️  环境超时，终止本轮")
                break

        # ── 结果统计 ──────────────────────────────────────────────────────
        success_info = env.unwrapped.get_success_info()
        n_ok         = sum(success_info.values())
        n_total      = len(success_info)
        all_ok       = n_ok == n_total

        print(f"\n    结果：{n_ok}/{n_total}  {'✅ PASS' if all_ok else '❌ FAIL'}")
        for obj_id, ok in success_info.items():
            print(f"      {'✅' if ok else '❌'}  {obj_id}")
        print(f"    步数：{step_count}")

        episode_results.append(all_ok)

        # ── 拼接视频帧 ────────────────────────────────────────────────────
        h = frame_h or 480
        w = frame_w or 640
        status_str  = "PASS" if all_ok else f"{n_ok}/{n_total}"
        title_text  = f"Episode {ep + 1}  [{perception_mode}]  {status_str}"
        all_frames += make_title_frames(title_text, h, w, n_frames=15)
        all_frames += ep_frames

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    n_success    = sum(episode_results)
    success_rate = n_success / n_episodes
    print("\n" + "═" * 60)
    print(f"  最终成功率（{perception_mode}）：")
    print(f"  {n_success}/{n_episodes} episodes = {success_rate:.0%}")
    print("═" * 60)

    if all_frames:
        out_path = ROOT / f"demo_{perception_mode}_all_episodes.mp4"
        imageio.mimsave(str(out_path), all_frames, fps=20)
        total_sec = len(all_frames) / 20
        print(f"  视频已保存：{out_path.name}  "
              f"({len(all_frames)} 帧 / {total_sec:.1f}s)")

    env.close()
    return success_rate


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modular Pick-and-Place Demo — runs full pipeline and reports results"
    )
    parser.add_argument(
        "--perception", type=str, default="state",
        choices=["state", "vision"],
        help="感知模式：state（特权状态）或 vision（视觉感知）",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="随机试验轮数（默认 10）",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="scene.yaml 路径（默认：config/scene.yaml）",
    )
    args = parser.parse_args()

    config = args.config or str(ROOT / "config" / "scene.yaml")
    run_pipeline(config, args.perception, args.episodes)