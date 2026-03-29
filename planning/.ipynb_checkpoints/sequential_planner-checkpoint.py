# planning/sequential_planner.py
from __future__ import annotations

import numpy as np
from typing import List

from perception.base import SceneRepresentation, ObjectInfo, TargetArea
from planning.base import BasePlanner, GraspAction, ActionSequence

# ── 常量 ──────────────────────────────────────────────────────────────────────
_LIFT_HEIGHT       = 0.15    # 抓起后抬升到桌面以上的高度（米）
_GRASP_DEPTH       = 0.027   # TCP 插入物体顶面以下的深度（米），确保稳定夹持
_SLOT_MARGIN       = 0.10    # 目标区域边距比例

# TCP 朝向：夹爪竖直朝下
# world_x → tcp_x:  [1, 0, 0]
# world_y → tcp_y:  [0,-1, 0]  (翻转，Panda TCP 约定)
# world_z → tcp_z:  [0, 0,-1]  (朝下)
_R_DOWN = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1],
], dtype=np.float64)


def _top_center_z(position_z: float, half_height: float) -> float:
    """
    物体顶面中心的 z 坐标。
    position_z : 物体质心 z（感知层统一语义）
    half_height: dimensions[2] / 2
    """
    return position_z + half_height


def _make_pose(xyz: np.ndarray) -> np.ndarray:
    """构造 TCP 竖直朝下的 4×4 位姿矩阵，平移为 xyz。"""
    T = np.eye(4)
    T[:3, :3] = _R_DOWN
    T[:3,  3] = xyz
    return T


class SequentialPlanner(BasePlanner):
    """
    顺序规划器。

    坐标语义约定（贯穿整个 planning 层）：
      - ObjectInfo.position   = 物体几何质心，世界坐标系
      - TCP 目标位置（抓取）  = 物体顶面中心 z − GRASP_DEPTH，朝下
      - TCP 目标位置（放置）  = slot 质心 z + 半高，朝下
        等价于：桌面 + 物体全高 − GRASP_DEPTH
      - slot.z（内部）        = 物体质心高度 = table_z + half_height
    """

    def __init__(self, lift_height: float = _LIFT_HEIGHT):
        self.lift_height = lift_height

    # ── 主入口 ────────────────────────────────────────────────────────────

    def plan(self, scene: SceneRepresentation) -> ActionSequence:
        ta      = scene.target_area
        objects = scene.objects

        slots = self._allocate_slots(objects, ta)
        order = self._sort_by_distance(objects, ta)

        actions: List[GraspAction] = []
        for idx in order:
            obj  = objects[idx]
            slot = slots[idx]          # slot = (x, y, z_centroid)

            actions.append(GraspAction(
                object_id   = obj.object_id,
                grasp_pose  = self._build_grasp_pose(obj),
                place_pose  = self._build_place_pose(slot, obj),
                lift_height = self.lift_height,
                object_dims = obj.dimensions.copy(),
            ))

        return ActionSequence(actions=actions, scene_repr=scene)

    # ── Slot 分配 ─────────────────────────────────────────────────────────

    def _allocate_slots(
        self,
        objects: List[ObjectInfo],
        ta: TargetArea,
    ) -> List[np.ndarray]:
        n    = len(objects)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    
        # 每个物体整体放入目标区域：可用边界需额外收缩物体 footprint 的一半
        # 取所有物体 XY 方向最大半径，作为统一的安全边距
        max_half_x = max(obj.dimensions[0] for obj in objects) / 2
        max_half_y = max(obj.dimensions[1] for obj in objects) / 2
    
        usable_w = ta.size[0] * (1 - _SLOT_MARGIN) - 2 * max_half_x
        usable_h = ta.size[1] * (1 - _SLOT_MARGIN) - 2 * max_half_y
    
        # 防止目标区域过小时 usable 变负
        usable_w = max(usable_w, 0.0)
        usable_h = max(usable_h, 0.0)
    
        # X 从大到小：远端先放，机械臂撤退时不扫过已放物体
        xs = np.linspace(
            ta.center[0] + usable_w / 2,
            ta.center[0] - usable_w / 2,
            cols,
        )
        ys = np.linspace(
            ta.center[1] - usable_h / 2,
            ta.center[1] + usable_h / 2,
            rows,
        )
    
        slots = []
        for i in range(n):
            r, c        = divmod(i, cols)
            x           = xs[c]
            y           = ys[r] if r < len(ys) else ys[-1]
            half_height = objects[i].dimensions[2] / 2
            z           = ta.table_z + half_height
            slots.append(np.array([x, y, z], dtype=np.float64))
    
        return slots

    # ── 排序：从远到近 ────────────────────────────────────────────────────

    def _sort_by_distance(
        self,
        objects: List[ObjectInfo],
        ta: TargetArea,
    ) -> List[int]:
        ta_xy     = ta.center                    # (2,)
        distances = [
            np.linalg.norm(obj.position[:2] - ta_xy)
            for obj in objects
        ]
        return np.argsort(distances)[::-1].tolist()

    # ── 构建抓取位姿 ──────────────────────────────────────────────────────

    def _build_grasp_pose(self, obj: ObjectInfo) -> np.ndarray:
        """
        TCP 目标 = 物体顶面中心，稍微深入 GRASP_DEPTH。

        pos[2] 是质心 z，顶面 z = pos[2] + half_height。
        再下移 GRASP_DEPTH，让夹爪微微夹入顶面保证稳定性。
        """
        half_height = obj.dimensions[2] / 2
        tcp_z       = _top_center_z(obj.position[2], half_height) - _GRASP_DEPTH
        xyz         = np.array([obj.position[0], obj.position[1], tcp_z])
        return _make_pose(xyz)

    # ── 构建放置位姿 ──────────────────────────────────────────────────────

    def _build_place_pose(
        self,
        slot: np.ndarray,
        obj: ObjectInfo,
    ) -> np.ndarray:
        """
        slot.z 是放置后物体质心高度。
        TCP 目标 = slot 顶面中心 − GRASP_DEPTH（与抓取逻辑完全对称）。
        """
        half_height = obj.dimensions[2] / 2
        tcp_z       = _top_center_z(slot[2], half_height) - _GRASP_DEPTH
        xyz         = np.array([slot[0], slot[1], tcp_z])
        return _make_pose(xyz)

    # ── 调试信息 ──────────────────────────────────────────────────────────

    def describe_plan(self, seq: ActionSequence) -> str:
        lines = [f"ActionSequence（共 {seq.n_actions} 步）："]
        ta = seq.scene_repr.target_area
        for i, act in enumerate(seq.actions):
            gp     = act.grasp_position.round(3)
            pp     = act.place_position.round(3)
            in_ta  = ta.contains(pp[:2])
            lines.append(
                f"  [{i+1}] {act.object_id}\n"
                f"        grasp  → ({gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f})\n"
                f"        place  → ({pp[0]:.3f}, {pp[1]:.3f}, {pp[2]:.3f})"
                f"  {'✅ 在目标区' if in_ta else '❌ 不在目标区'}"
            )
        return "\n".join(lines)