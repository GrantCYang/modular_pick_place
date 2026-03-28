# execution/motion_executor.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch

from planning.base import ActionSequence, GraspAction


class Phase(Enum):
    OPEN_GRIPPER_INIT = auto()
    PRE_GRASP         = auto()
    GRASP_DESCEND     = auto()
    CLOSE_GRIPPER     = auto()
    LIFT              = auto()
    TRANSPORT         = auto()
    PLACE_DESCEND     = auto()
    OPEN_GRIPPER      = auto()
    RETREAT           = auto()
    DONE              = auto()


@dataclass
class ExecutorConfig:
    pre_grasp_height:  float = 0.10
    lift_height:       float = 0.15
    place_clearance:   float = 0.05
    retreat_height:    float = 0.15

    move_speed:        float = 0.1
    pos_threshold:     float = 0.02
    gripper_wait:      int   = 15
    max_steps_phase:   int   = 1000

    gripper_open:      float =  1.0
    gripper_close:     float = -1.0


@dataclass
class ExecutionState:
    phase:         Phase = Phase.OPEN_GRIPPER_INIT
    phase_step:    int   = 0
    action_index:  int   = 0
    total_steps:   int   = 0


class MotionExecutor:
    """
    控制器：pd_ee_delta_pose
      arm    6维：[dx, dy, dz, drx, dry, drz]  — 位置增量(m) + 旋转增量(轴角,rad)
      gripper 1维：+1 张开，-1 夹紧
      合计  7维，且必须以 batched tensor shape=(1,7) 传入 env.step()

    pos_action_scale 默认 0.1：action[0:3]=1.0 对应 0.1m 位移
    rot_action_scale 默认 0.1：action[3:6]=1.0 对应 0.1rad 旋转
    """

    # ── 控制器参数（从 ManiSkill 源码确认） ─────────────────────────────────
    POS_ACTION_SCALE = 0.1   # action=1 → 0.1m
    ACTION_DIM       = 7     # 6(arm) + 1(gripper)

    def __init__(self, env, config: Optional[ExecutorConfig] = None):
        self._env   = env
        self.cfg    = config or ExecutorConfig()
        self._seq:  Optional[ActionSequence] = None
        self._state = ExecutionState()

        # 确认环境用的 device（cpu or cuda）
        p = self._env.unwrapped.agent.tcp.pose.p
        self._device = p.device if isinstance(p, torch.Tensor) else torch.device("cpu")

    # ── 外部接口 ──────────────────────────────────────────────────────────

    def load(self, seq: ActionSequence) -> None:
        self._seq   = seq
        self._state = ExecutionState()

    def is_done(self) -> bool:
        if self._seq is None:
            return True
        return self._state.action_index >= self._seq.n_actions

    def step(self) -> torch.Tensor:
        """
        返回 shape=(1, 7) 的 torch.Tensor（float32），
        可直接传入 env.step()。
        """
        if self.is_done():
            return self._to_tensor(self._idle_action())

        cur_action = self._seq.actions[self._state.action_index]
        action_np  = self._execute_phase(cur_action)

        self._state.total_steps += 1
        self._state.phase_step  += 1
        return self._to_tensor(action_np)

    @property
    def current_phase(self) -> Phase:
        return self._state.phase

    @property
    def current_action_index(self) -> int:
        return self._state.action_index

    # ── 阶段状态机 ────────────────────────────────────────────────────────

    def _execute_phase(self, grasp_action: GraspAction) -> np.ndarray:
        phase = self._state.phase
        cfg   = self.cfg

        if phase == Phase.OPEN_GRIPPER_INIT:
            action = self._gripper_action(cfg.gripper_open)
            if self._state.phase_step >= cfg.gripper_wait:
                self._next_phase()
            return action

        if phase == Phase.PRE_GRASP:
            gp = grasp_action.grasp_position
            target = np.array([gp[0], gp[1], gp[2] + cfg.pre_grasp_height])
            action = self._move_to(target, gripper=cfg.gripper_open)
            if self._reached(target) or self._phase_timeout():
                self._next_phase()
            return action

        if phase == Phase.GRASP_DESCEND:
            target = grasp_action.grasp_position.copy()
            action = self._move_to(target, gripper=cfg.gripper_open)
            if self._reached(target) or self._phase_timeout():
                self._next_phase()
            return action

        if phase == Phase.CLOSE_GRIPPER:
            action = self._gripper_action(cfg.gripper_close)
            if self._state.phase_step >= cfg.gripper_wait:
                self._next_phase()
            return action

        if phase == Phase.LIFT:
            tcp_xy = self._get_tcp_pos()[:2]
            target = np.array([tcp_xy[0], tcp_xy[1], cfg.lift_height])
            action = self._move_to(target, gripper=cfg.gripper_close)
            if self._reached(target) or self._phase_timeout():
                self._next_phase()
            return action

        if phase == Phase.TRANSPORT:
            pp = grasp_action.place_position
            target = np.array([pp[0], pp[1], cfg.lift_height])
            action = self._move_to(target, gripper=cfg.gripper_close)
            if self._reached(target) or self._phase_timeout():
                self._next_phase()
            return action

        if phase == Phase.PLACE_DESCEND:
            target = grasp_action.place_position.copy()
            action = self._move_to(target, gripper=cfg.gripper_close)
            if self._reached(target) or self._phase_timeout():
                self._next_phase()
            return action

        if phase == Phase.OPEN_GRIPPER:
            action = self._gripper_action(cfg.gripper_open)
            if self._state.phase_step >= cfg.gripper_wait:
                self._next_phase()
            return action

        if phase == Phase.RETREAT:
            tcp_xy = self._get_tcp_pos()[:2]
            target = np.array([tcp_xy[0], tcp_xy[1], cfg.retreat_height])
            action = self._move_to(target, gripper=cfg.gripper_open)
            if self._reached(target) or self._phase_timeout():
                self._finish_current_action()
            return action

        return self._idle_action()

    # ── 阶段推进 ──────────────────────────────────────────────────────────

    _PHASE_ORDER = [
        Phase.OPEN_GRIPPER_INIT,
        Phase.PRE_GRASP,
        Phase.GRASP_DESCEND,
        Phase.CLOSE_GRIPPER,
        Phase.LIFT,
        Phase.TRANSPORT,
        Phase.PLACE_DESCEND,
        Phase.OPEN_GRIPPER,
        Phase.RETREAT,
        Phase.DONE,
    ]

    def _next_phase(self) -> None:
        cur_idx = self._PHASE_ORDER.index(self._state.phase)
        self._state.phase      = self._PHASE_ORDER[cur_idx + 1]
        self._state.phase_step = 0

    def _finish_current_action(self) -> None:
        self._state.action_index += 1
        self._state.phase        = Phase.OPEN_GRIPPER_INIT
        self._state.phase_step   = 0

    def _phase_timeout(self) -> bool:
        return self._state.phase_step >= self.cfg.max_steps_phase

    # ── action 构建（7维 numpy） ────────────────────────────────────────

    def _move_to(self, target: np.ndarray, gripper: float) -> np.ndarray:
        """
        计算朝 target 移动一步的 7 维 action。
        action[0:3] = delta_xyz / POS_ACTION_SCALE，clip 到 [-1, 1]
        action[3:6] = 0（保持当前姿态，不旋转）
        action[6]   = gripper
        """
        tcp_pos = self._get_tcp_pos()
        delta   = target - tcp_pos

        # 按速度上限裁剪
        dist = np.linalg.norm(delta)
        if dist > self.cfg.move_speed:
            delta = delta / dist * self.cfg.move_speed

        action_xyz = np.clip(delta / self.POS_ACTION_SCALE, -1.0, 1.0)

        action      = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action[0:3] = action_xyz   # dx, dy, dz
        action[3:6] = 0.0          # drx, dry, drz（不旋转）
        action[6]   = gripper      # gripper
        return action

    def _gripper_action(self, gripper_val: float) -> np.ndarray:
        """原地不动，只控制夹爪"""
        action    = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action[6] = gripper_val
        return action

    def _idle_action(self) -> np.ndarray:
        action    = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action[6] = self.cfg.gripper_open
        return action

    # ── tensor 转换：numpy (7,) → torch (1, 7) ────────────────────────

    def _to_tensor(self, action_np: np.ndarray) -> torch.Tensor:
        """
        env.step() 期望 shape=(1, 7) 的 batched tensor。
        """
        return torch.tensor(
            action_np[None, :],          # (7,) → (1, 7)
            dtype=torch.float32,
            device=self._device,
        )

    # ── TCP 位置读取 ────────────────────────────────────────────────────

    def _get_tcp_pos(self) -> np.ndarray:
        """
        返回 TCP 世界坐标 (x, y, z)，shape (3,) numpy float64。
        env.unwrapped.agent.tcp.pose.p  shape: (num_envs, 3)
        """
        p = self._env.unwrapped.agent.tcp.pose.p
        if isinstance(p, torch.Tensor):
            return p[0].cpu().numpy().astype(np.float64)
        return np.array(p[0], dtype=np.float64)

    def _reached(self, target: np.ndarray) -> bool:
        return np.linalg.norm(self._get_tcp_pos() - target) < self.cfg.pos_threshold