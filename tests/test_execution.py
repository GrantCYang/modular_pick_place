# tests/test_execution.py
"""
执行层测试：
  Step 1 — 单阶段冒烟测试：只跑 PRE_GRASP 阶段，确认 TCP 能到达目标位置
  Step 2 — 单个 GraspAction 完整流程：跑完一个物体的 pick-and-place，录制视频
  Step 3 — 完整 Episode：跑完所有物体，统计成功率

用法：
    python tests/test_execution.py --step 1
    python tests/test_execution.py --step 2
    python tests/test_execution.py --step 3   ← 完整流程
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio
import torch
import gymnasium as gym

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                            # noqa
from envs.multi_object_env import MultiObjectPickAndPlaceEnv      # noqa
from perception.state_perception import StatePerception
from planning.sequential_planner import SequentialPlanner
from execution.motion_executor import MotionExecutor, ExecutorConfig, Phase


# ── 工具 ──────────────────────────────────────────────────────────────────────

def tensor_to_uint8(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return frame


def make_env(config_path: str) -> gym.Env:
    return gym.make(
        "MultiObjectPickAndPlace-v1",
        render_mode="rgb_array",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        scene_config=config_path,
    )


def get_tcp_pos(env) -> np.ndarray:
    p = env.unwrapped.agent.tcp.pose.p
    if isinstance(p, torch.Tensor):
        return p[0].cpu().numpy()
    return np.array(p[0])


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1：单阶段冒烟测试
#  验证 _move_to 能让 TCP 到达指定点，与物体无关
# ─────────────────────────────────────────────────────────────────────────────

def test_step1_smoke(config_path: str):
    print("\n" + "═"*55)
    print("  Step 1 — 单阶段冒烟测试")
    print("═"*55)

    env = make_env(config_path)
    obs, _ = env.reset(seed=0)

    # 确认 action space
    print(f"  action_space: {env.action_space}")   # 应该是 (1, 7)
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
        tcp  = get_tcp_pos(env)
        delta = TARGET - tcp
        dist  = np.linalg.norm(delta)
        if dist > MOVE_SPEED:
            delta = delta / dist * MOVE_SPEED

        # ✅ 7维，batched (1, 7) tensor
        action_np         = np.zeros(7, dtype=np.float32)
        action_np[0:3]    = np.clip(delta / POS_SCALE, -1.0, 1.0)
        action_np[6]      = 1.0   # 夹爪张开
        action_tensor     = torch.tensor(
            action_np[None, :], dtype=torch.float32, device=device
        )

        obs, _, terminated, truncated, _ = env.step(action_tensor)
        frames.append(tensor_to_uint8(env.render()))

        if step % 30 == 0:
            pos  = get_tcp_pos(env)
            print(f"  step={step:3d}  TCP={pos.round(3)}  dist={np.linalg.norm(pos-TARGET):.4f}")

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
#  只执行 ActionSequence 中的第一个 GraspAction
# ─────────────────────────────────────────────────────────────────────────────

def test_step2_single(config_path: str):
    print("\n" + "═"*55)
    print("  Step 2 — 单个物体完整 pick-and-place")
    print("═"*55)

    env        = make_env(config_path)
    perception = StatePerception(env)
    planner    = SequentialPlanner()
    executor   = MotionExecutor(env)

    obs, _ = env.reset(seed=0)
    scene  = perception.observe(obs)
    seq    = planner.plan(scene)

    # 只执行第一个 GraspAction
    from planning.base import ActionSequence
    single_seq = ActionSequence(
        actions=[seq.actions[0]],
        scene_repr=scene,
    )
    executor.load(single_seq)

    target_obj = seq.actions[0].object_id
    print(f"  目标物体：{target_obj}")
    print(f"  grasp_pos：{seq.actions[0].grasp_position.round(3)}")
    print(f"  place_pos：{seq.actions[0].place_position.round(3)}")

    frames = []
    phase_log = []
    prev_phase = None
    step_count = 0

    while not executor.is_done():
        cur_phase = executor.current_phase
        if cur_phase != prev_phase:
            print(f"\n  ▶ 进入阶段：{cur_phase.name}")
            print(f"    TCP 当前位置：{get_tcp_pos(env).round(3)}")
            phase_log.append((executor._state.total_steps, cur_phase.name))
            prev_phase = cur_phase

        action = executor.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(tensor_to_uint8(env.render()))

        step_count += 1
        if terminated or truncated:
            print("  ⚠️  Episode 提前结束")
            break

    print(f"\n  最终 TCP 位置：{get_tcp_pos(env).round(3)}")
    print(f"    总步数: {step_count}")
    success = env.unwrapped.get_success_info()
    print(f"  目标物体到位：{'✅' if success.get(target_obj, False) else '❌'}")

    imageio.mimsave("debug_exec_step2.mp4", frames, fps=20)
    print(f"  视频已保存：debug_exec_step2.mp4  ({len(frames)} 帧)")
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3：完整 Episode（所有物体）
# ─────────────────────────────────────────────────────────────────────────────

def test_step3_full(config_path: str, n_episodes: int = 3):
    print("\n" + "═"*55)
    print(f"  Step 3 — 完整 Episode × {n_episodes}")
    print("═"*55)

    env        = make_env(config_path)
    perception = StatePerception(env)
    planner    = SequentialPlanner()
    cfg        = ExecutorConfig()
    executor   = MotionExecutor(env, cfg)

    episode_results = []

    for ep in range(n_episodes):
        print(f"\n  ── Episode {ep+1}/{n_episodes} ──")
        obs, _ = env.reset(seed=ep * 13)

        scene = perception.observe(obs)
        seq   = planner.plan(scene)
        executor.load(seq)

        frames     = []
        prev_phase = None
        step_count = 0
        max_steps  = cfg.max_steps_phase * 9 * seq.n_actions + 50

        while not executor.is_done() and step_count < max_steps:
            cur_phase = executor.current_phase
            if cur_phase != prev_phase:
                obj_idx = executor.current_action_index
                obj_id  = seq.actions[obj_idx].object_id if obj_idx < seq.n_actions else "—"
                print(f"    [{obj_id}] {cur_phase.name}")
                prev_phase = cur_phase

            action = executor.step()
            obs, reward, terminated, truncated, _ = env.step(action)

            if ep == 0:   # 只录第一个 episode
                frames.append(tensor_to_uint8(env.render()))

            step_count += 1
            if terminated or truncated:
                break

        # 检查结果
        success_info = env.unwrapped.get_success_info()
        all_ok       = all(success_info.values())
        n_ok         = sum(success_info.values())
        n_total      = len(success_info)

        print(f"\n    成功: {n_ok}/{n_total}  {'✅' if all_ok else '❌'}")
        for obj_id, ok in success_info.items():
            print(f"    {'✅' if ok else '❌'}  {obj_id}")
        print(f"    总步数: {step_count}")

        episode_results.append(all_ok)

        if ep == 0 and frames:
            video_path = "debug_exec_step3_ep0.mp4"
            imageio.mimsave(video_path, frames, fps=20)
            print(f"    视频已保存：{video_path}")

        print(f"  max_steps={max_steps}, n_actions={seq.n_actions}")

        # if terminated or truncated:
        #     print(f"    ⚠️ 提前结束：terminated={terminated}, truncated={truncated}, step={step_count}")
        #     break

    success_rate = sum(episode_results) / len(episode_results)
    print(f"\n  总成功率：{sum(episode_results)}/{len(episode_results)} = {success_rate:.0%}")
    env.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",   type=int, default=3,
                        help="1=冒烟, 2=单物体, 3=完整episode")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    config = args.config or str(ROOT / "config" / "scene.yaml")

    if args.step >= 1:
        test_step1_smoke(config)
    if args.step >= 2:
        test_step2_single(config)
    if args.step >= 3:
        test_step3_full(config, args.episodes)