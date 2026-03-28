# perception/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────
#  数据类：感知层输出（也是规划层唯一输入）
# ─────────────────────────────────────────────

@dataclass
class ObjectInfo:
    """
    单个物体的结构化描述。
    pose 使用 4×4 齐次变换矩阵，世界坐标系下。
    dimensions 为物体的 AABB 包围盒尺寸 (x, y, z)，单位：米。
    """
    object_id: str                        # 唯一标识符，如 "cube_0"、"cylinder_1"
    category: str                         # 物体类别，如 "box"、"cylinder"、"mesh"
    pose: np.ndarray                      # shape (4, 4)，SE3 齐次矩阵
    dimensions: np.ndarray                # shape (3,)，(length, width, height)
    is_grasped: bool = False              # 当前是否被夹爪持有（closed-loop 时更新）
    confidence: float = 1.0              # 感知置信度（state-based 时固定为 1.0）

    def __post_init__(self):
        assert self.pose.shape == (4, 4), "pose must be a 4x4 homogeneous matrix"
        assert self.dimensions.shape == (3,), "dimensions must be (x, y, z)"

    @property
    def position(self) -> np.ndarray:
        """物体中心在世界坐标系下的 (x, y, z)"""
        return self.pose[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """旋转矩阵，shape (3, 3)"""
        return self.pose[:3, :3]


@dataclass
class TargetArea:
    """
    目标放置区域的描述。
    center：区域中心在世界坐标系下的 (x, y) 坐标，z 由桌面高度决定。
    size：区域尺寸 (width_x, width_y)，单位：米。
    table_z：桌面高度，用于计算放置的 z 坐标。
    """
    center: np.ndarray        # shape (2,)，(x, y)
    size: np.ndarray          # shape (2,)，(width_x, width_y)
    table_z: float            # 桌面 z 高度，单位：米

    def __post_init__(self):
        assert self.center.shape == (2,), "center must be (x, y)"
        assert self.size.shape == (2,), "size must be (width_x, width_y)"

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        """判断给定 (x, y) 是否在目标区域内（支持 margin 缩进）"""
        half = self.size / 2.0 - margin
        return bool(np.all(np.abs(xy - self.center) <= half))

    def sample_placement_positions(self, n: int) -> List[np.ndarray]:
        """
        在目标区域内均匀采样 n 个放置坐标 (x, y, z)。
        以网格方式排列，防止物体叠放。
        """
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        xs = np.linspace(
            self.center[0] - self.size[0] / 2 * 0.8,
            self.center[0] + self.size[0] / 2 * 0.8,
            cols
        )
        ys = np.linspace(
            self.center[1] - self.size[1] / 2 * 0.8,
            self.center[1] + self.size[1] / 2 * 0.8,
            rows
        )
        positions = []
        for i in range(n):
            r, c = divmod(i, cols)
            positions.append(np.array([xs[c], ys[r], self.table_z]))
        return positions


@dataclass
class SceneRepresentation:
    """
    感知模块的输出，也是规划模块的完整输入。
    这是三个模块之间最核心的数据契约。
    规划和执行模块只能通过这个类了解场景，永远不接触原始 obs。
    """
    objects: List[ObjectInfo]             # 场景中所有待处理物体
    target_area: TargetArea               # 目标放置区域

    # 元信息，不参与规划逻辑，仅用于 debug/logging
    timestamp: float = 0.0               # 感知时刻（环境 step 数）
    perception_mode: str = "unknown"     # "state" 或 "vision"

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def get_object_by_id(self, object_id: str) -> Optional[ObjectInfo]:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def __repr__(self) -> str:
        obj_ids = [o.object_id for o in self.objects]
        return (
            f"SceneRepresentation("
            f"objects={obj_ids}, "
            f"target_center={self.target_area.center}, "
            f"mode={self.perception_mode})"
        )


# ─────────────────────────────────────────────
#  接口：所有感知模块必须实现这个 ABC
# ─────────────────────────────────────────────

class BasePerception(ABC):
    """
    感知模块抽象基类。
    子类只需实现 observe()，接收原始 obs dict，返回 SceneRepresentation。
    任何感知实现（state-based、vision-based、网络推理）都遵循同一接口。
    """

    @abstractmethod
    def observe(self, obs: dict) -> SceneRepresentation:
        """
        Args:
            obs: 来自 env.step() 或 env.reset() 的原始观测字典
        Returns:
            SceneRepresentation: 结构化场景描述，供 Planning 模块消费
        """
        ...

    def reset(self) -> None:
        """episode 开始时重置感知模块内部状态（如滤波器、历史帧等）。默认无操作。"""
        pass