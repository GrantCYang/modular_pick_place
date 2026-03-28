# planning/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from perception.base import SceneRepresentation


# ── 单步动作：抓一个物体并放到指定位置 ────────────────────────────────────────

@dataclass
class GraspAction:
    """
    对单个物体的完整 pick-and-place 描述。
    规划层输出这个，执行层消费这个，两者通过这个结构解耦。
    所有位姿均为世界坐标系下的 4×4 齐次矩阵。
    """
    object_id:    str           # 对应 ObjectInfo.object_id
    grasp_pose:   np.ndarray    # shape (4,4)：夹爪闭合时的目标末端位姿
    place_pose:   np.ndarray    # shape (4,4)：放置时的目标末端位姿
    lift_height:  float = 0.15  # 抓取后抬升高度（相对桌面，单位：米）
    object_dims:  Optional[np.ndarray] = None  # shape (3,)，用于执行层计算接近高度

    def __post_init__(self):
        assert self.grasp_pose.shape == (4, 4)
        assert self.place_pose.shape == (4, 4)

    @property
    def grasp_position(self) -> np.ndarray:
        return self.grasp_pose[:3, 3]

    @property
    def place_position(self) -> np.ndarray:
        return self.place_pose[:3, 3]


# ── 完整的动作序列：规划层的完整输出 ──────────────────────────────────────────

@dataclass
class ActionSequence:
    """
    规划层的完整输出，是一个有序的 GraspAction 列表。
    执行层按顺序执行，不跳过，不重排。
    """
    actions:    List[GraspAction]
    scene_repr: SceneRepresentation   # 生成该序列时所用的场景快照，用于 debug

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        ids = [a.object_id for a in self.actions]
        return f"ActionSequence(order={ids}, n={self.n_actions})"


# ── 规划器接口 ─────────────────────────────────────────────────────────────────

class BasePlanner(ABC):
    """
    规划模块抽象基类。
    输入：SceneRepresentation（感知层输出）
    输出：ActionSequence（执行层输入）
    """

    @abstractmethod
    def plan(self, scene: SceneRepresentation) -> ActionSequence:
        """
        给定当前场景描述，生成完整的动作序列。
        """
        ...

    def reset(self) -> None:
        """每次 episode 开始时重置规划器内部状态（如有）。"""
        pass