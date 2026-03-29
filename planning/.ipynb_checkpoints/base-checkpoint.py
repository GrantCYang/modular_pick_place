# planning/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from perception.base import SceneRepresentation


@dataclass
class GraspAction:
    """
    对单个物体的完整 pick-and-place 描述。
    所有位姿均为世界坐标系下的 4×4 齐次矩阵。
    位置语义统一：TCP 目标位置 = 物体顶面中心（质心 + 半高），朝向竖直朝下。
    """
    object_id:   str
    grasp_pose:  np.ndarray          # shape (4,4)
    place_pose:  np.ndarray          # shape (4,4)
    lift_height: float = 0.15        # 抓取后抬升到桌面以上的高度，单位：米
    object_dims: Optional[np.ndarray] = None   # shape (3,)

    def __post_init__(self):
        assert self.grasp_pose.shape == (4, 4)
        assert self.place_pose.shape == (4, 4)

    @property
    def grasp_position(self) -> np.ndarray:
        return self.grasp_pose[:3, 3]

    @property
    def place_position(self) -> np.ndarray:
        return self.place_pose[:3, 3]


@dataclass
class ActionSequence:
    actions:    List[GraspAction]
    scene_repr: SceneRepresentation

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def __repr__(self) -> str:
        ids = [a.object_id for a in self.actions]
        return f"ActionSequence(order={ids}, n={self.n_actions})"


class BasePlanner(ABC):

    @abstractmethod
    def plan(self, scene: SceneRepresentation) -> ActionSequence:
        ...

    def reset(self) -> None:
        pass