# tests/test_env.py
"""
环境冒烟测试：
  - 检查物体数量是否正确（无幽灵 cube）
  - 保存初始帧截图 debug_env_init.png
  - 随机动作 50 步并保存视频 debug_env_random_run.mp4
  - 打印特权状态，确认物体位置合法

用法：
    python tests/test_env.py
    python tests/test_env.py --config config/scene.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import imageio
import torch
import gymnasium as gym

# ── 确保项目根目录在 sys.path 里 ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                   # noqa: F401，触发内置环境注册
from envs.multi_object_env import MultiObjectPickAndPlaceEnv  # noqa: F401，触发自定义环境注册


# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_uint8(frame) -> np.ndarray:
    """
    将 ManiSkill 返回的帧（可能是 torch tensor / numpy / 带 batch 维度）
    统一转为 (H, W, 3) uint8 numpy 数组。
    """
    # torch tensor → numpy
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()

    # 去掉 batch 维度 (1, H, W, 3) → (H, W, 3)
    if frame.ndim == 4:
        frame = frame[0]

    # float [0,1] → uint8 [0,255]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

    return frame


def check_objects(env, expected_count: int):
    """断言场景中物体数量正确（无幽灵 cube）"""
    actual = len(env.unwrapped.objects)
    status = "✅" if actual == expected_count else "❌"
    print(f"{status} 物体数量：期望 {expected_count}，实际 {actual}")
    if actual != expected_count:
        print("   ⚠️  数量不符！请检查是否仍有父类幽灵 cube 残留。")
    return actual == expected_count


def print_privileged_state(env):
    """打印特权状态，并做简单的合法性检查"""
    state = env.unwrapped.get_privileged_state()
    ta = state["target"]

    print(f"\n📦 特权状态 — 检测到 {len(state['objects'])} 个物体：")
    for obj in state["objects"]:
        pos = obj["pose"][:3, 3].round(3)
        xy  = pos[:2]

        # 检查是否在桌面生成范围内
        x_ok = -0.20 <= xy[0] <= 0.10
        y_ok = -0.20 <= xy[1] <= 0.20
        flag = "✅" if (x_ok and y_ok) else "⚠️ 超出范围"

        print(f"  {flag}  [{obj['id']}] category={obj['category']}, "
              f"pos={pos}, dims={obj['dimensions'].round(3)}")

    print(f"\n🎯 目标区域：center={ta['center']}, "
          f"size={ta['size']}, table_z={ta['table_z']}")

    return state


def run_test(config_path: str | None = None,
             n_steps: int = 50,
             fps: int = 20,
             out_image: str = "debug_env_init.png",
             out_video: str = "debug_env_random_run.mp4"):

    config_path = config_path or str(ROOT / "config" / "scene.yaml")
    print(f"🚀 测试启动")
    print(f"   配置文件  : {config_path}")
    print(f"   随机步数  : {n_steps}")
    print(f"   输出图片  : {out_image}")
    print(f"   输出视频  : {out_video}")
    print()

    # ── 1. 创建环境 ────────────────────────────────────────────────────────
    # render_mode="rgb_array" → env.render() 走 _default_human_render_camera_configs
    # obs_mode="state"        → obs 是关节状态向量，不含图像（图像由 render() 单独获取）
    env = gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode="rgb_array",
        obs_mode="state",
        scene_config=config_path,       # ✅ 新版参数名
    )
    print(f"✅ 环境创建成功")
    print(f"   obs space  : {env.observation_space}")
    print(f"   act space  : {env.action_space}")

    # ── 2. Reset ───────────────────────────────────────────────────────────
    obs, info = env.reset(seed=42)
    print(f"\n✅ env.reset() 完成，obs shape = {obs.shape}")

    # ── 3. 物体数量检查 ────────────────────────────────────────────────────
    expected = len(env.unwrapped.scene_cfg.objects)
    check_objects(env, expected)

    # ── 4. 保存初始帧 ──────────────────────────────────────────────────────
    raw_frame = env.render()
    init_frame = tensor_to_uint8(raw_frame)
    imageio.imwrite(out_image, init_frame)
    print(f"\n📸 初始帧已保存：{out_image}  (shape={init_frame.shape})")

    # ── 5. 打印特权状态 ────────────────────────────────────────────────────
    print_privileged_state(env)

    # ── 6. 随机动作 + 录制视频 ─────────────────────────────────────────────
    print(f"\n⚙️  运行 {n_steps} 步随机动作并录制视频...")
    frames = [init_frame]   # 第一帧已经有了

    total_reward = 0.0
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # reward 可能是 tensor
        if isinstance(reward, torch.Tensor):
            total_reward += reward.mean().item()
        else:
            total_reward += float(reward)

        frame = tensor_to_uint8(env.render())
        frames.append(frame)

        if terminated or truncated:
            print(f"   Episode 在第 {step+1} 步结束，重置环境...")
            obs, info = env.reset()

    print(f"   平均奖励：{total_reward / n_steps:.4f}")

    # ── 7. 保存视频 ────────────────────────────────────────────────────────
    imageio.mimsave(out_video, frames, fps=fps)
    print(f"\n🎬 视频已保存：{out_video}  ({len(frames)} 帧, {fps}fps)")

    # ── 8. 成功率统计 ──────────────────────────────────────────────────────
    success_info = env.unwrapped.get_success_info()
    print(f"\n📊 当前 Episode 成功状态：")
    for obj_id, done in success_info.items():
        print(f"   {'✅' if done else '❌'}  {obj_id}")

    env.close()
    print("\n🏁 测试完成！")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiObjectPickAndPlace 环境测试")
    parser.add_argument(
        "--config", type=str, default=None,
        help="scene.yaml 路径，默认使用 config/scene.yaml"
    )
    parser.add_argument("--steps",  type=int, default=50,  help="随机动作步数")
    parser.add_argument("--fps",    type=int, default=20,  help="视频帧率")
    parser.add_argument("--image",  type=str, default="debug_env_init.png")
    parser.add_argument("--video",  type=str, default="debug_env_random_run.mp4")
    args = parser.parse_args()

    run_test(
        config_path=args.config,
        n_steps=args.steps,
        fps=args.fps,
        out_image=args.image,
        out_video=args.video,
    )