# planning/sequential_planner.py
from __future__ import annotations

import numpy as np
from typing import List, Tuple

from perception.base import SceneRepresentation, ObjectInfo, TargetArea
from planning.base import BasePlanner, GraspAction, ActionSequence


# TCP 在物体正上方的默认接近偏移（沿世界 Z 轴）
_PRE_GRASP_CLEARANCE = 0.10   # 10cm：下降前停在物体上方这个高度
_LIFT_HEIGHT         = 0.15   # 15cm：抓起后抬升到桌面以上这个高度
_PLACE_CLEARANCE     = 0.05   # 5cm：放置前停在目标位置上方这个高度


class SequentialPlanner(BasePlanner):
    """
    顺序规划器：为每个物体生成一个 GraspAction，按距离目标区域从远到近排序。

    策略：
      1. Slot 分配：把目标区域划分为 N 个不重叠的放置格，每个物体分配一个
      2. 抓取排序：距离目标区域最远的物体最先被抓（减少后续运动中的遮挡风险）
      3. 放置顺序：配合排序，先放在目标区域远端（X 大）的 slot，
                  机械臂撤退时不会扫过已放物体
    """

    def __init__(self, lift_height: float = _LIFT_HEIGHT):
        self.lift_height = lift_height

    # ── 主入口 ─────────────────────────────────────────────────────────────

    def plan(self, scene: SceneRepresentation) -> ActionSequence:
        ta = scene.target_area
        objects = scene.objects

        # ① 为每个物体分配一个目标 slot（考虑物体尺寸，避免重叠）
        slots = self._allocate_slots(objects, ta)

        # ② 按物体到目标区域的距离从远到近排序
        order = self._sort_by_distance(objects, ta)

        # ③ 按排序顺序构建 GraspAction 列表
        actions: List[GraspAction] = []
        for rank, idx in enumerate(order):
            obj  = objects[idx]
            slot = slots[idx]   # slot 是 (x, y, z) 世界坐标

            grasp_pose = self._build_grasp_pose(obj)
            place_pose = self._build_place_pose(slot, obj)

            actions.append(GraspAction(
                object_id   = obj.object_id,
                grasp_pose  = grasp_pose,
                place_pose  = place_pose,
                lift_height = self.lift_height,
                object_dims = obj.dimensions.copy(),
            ))

        return ActionSequence(actions=actions, scene_repr=scene)

    # ── Slot 分配 ──────────────────────────────────────────────────────────

    def _allocate_slots(
        self,
        objects: List[ObjectInfo],
        ta: TargetArea,
    ) -> List[np.ndarray]:
        """
        在目标区域内为 N 个物体分配不重叠的放置格。

        布局算法：
          - 计算每个物体在 XY 平面的最大占地半径
          - 用网格布局：先算行列数，再均匀分配
          - 留 10% 边距避免物体压边
        """
        n = len(objects)

        # 每个物体的"占地半径"（取 XY 最大半边长 + 5mm 间隙）
        footprints = [
            max(obj.dimensions[0], obj.dimensions[1]) / 2 + 0.005
            for obj in objects
        ]

        # 网格列数：尽量接近正方形
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # 可用区域（留边距）
        margin = 0.10   # 保留 10% 边距
        usable_w = ta.size[0] * (1 - margin)
        usable_h = ta.size[1] * (1 - margin)

        # 网格节点坐标（中心对齐到 ta.center）
        xs = np.linspace(
            ta.center[0] - usable_w / 2,
            ta.center[0] + usable_w / 2,
            cols,
        )
        ys = np.linspace(
            ta.center[1] - usable_h / 2,
            ta.center[1] + usable_h / 2,
            rows,
        )

        # 先按 X 从大到小排列 slot（远端先放，机械臂不会扫过已放物体）
        xs = xs[::-1]   # X 大 → 远离机器人（机器人在 X 负方向）

        slots = []
        for i in range(n):
            r, c = divmod(i, cols)
            x = xs[c]
            y = ys[r] if r < len(ys) else ys[-1]
            z = ta.table_z + objects[i].dimensions[2] / 2   # 物体半高，放置后底面贴桌
            slots.append(np.array([x, y, z]))

        return slots

    # ── 排序：从远到近 ────────────────────────────────────────────────────

    def _sort_by_distance(
        self,
        objects: List[ObjectInfo],
        ta: TargetArea,
    ) -> List[int]:
        """
        返回物体索引列表，按物体中心到目标区域中心的距离从远到近排列。
        最远的物体最先被抓取。
        """
        ta_center_3d = np.array([ta.center[0], ta.center[1], 0.0])
        distances = [
            np.linalg.norm(obj.position - ta_center_3d)
            for obj in objects
        ]
        # argsort 升序，reverse 得到从远到近
        order = np.argsort(distances)[::-1].tolist()
        return order

    # ── 构建抓取位姿 ───────────────────────────────────────────────────────

    def _build_grasp_pose(self, obj: ObjectInfo) -> np.ndarray:
        """
        构建抓取时末端（TCP）的目标位姿。

        约定：
          - TCP 中心对准物体中心（XY），Z = 物体顶面（让夹爪从顶部夹住）
          - 夹爪朝下（世界 Z 轴负方向），即末端 Z 轴 = [0,0,-1] in world frame
          - 对于圆柱/方块，旋转固定为 "夹爪竖直朝下" 姿态

        注意：这里的旋转矩阵表示的是 TCP 坐标系在世界坐标系下的朝向。
        ManiSkill Panda 的 TCP 默认 Z 轴朝下（即 [0,0,-1] 是夹爪闭合方向）。
        所以 "夹爪朝下" 对应 TCP Z 轴 = 世界 Z 轴负方向，
        也就是旋转矩阵 = diag([-1, 1, -1])（绕 Y 轴旋转 180°）。
        """
        pos = obj.position.copy()

        # Z：夹爪中心对准物体顶面，稍微深入一点确保稳定抓取
        grasp_z = pos[2] - obj.dimensions[2] * 0.1   # 顶面下 1cm

        # 旋转：TCP Z 轴朝下（夹爪竖直下压）
        # world_x → tcp_x: [1,0,0]（保持朝 X 正方向）
        # world_y → tcp_y: [0,-1,0]（翻转）
        # world_z → tcp_z: [0,0,-1]（朝下）
        # 即 R = Ry(180°)
        R = np.array([
            [ 1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1],
        ], dtype=np.float64)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [pos[0], pos[1], grasp_z]
        return T

    # ── 构建放置位姿 ───────────────────────────────────────────────────────

    def _build_place_pose(
        self,
        slot: np.ndarray,
        obj: ObjectInfo,
    ) -> np.ndarray:
        """
        构建放置时末端（TCP）的目标位姿。
        slot 是 (x, y, z)，z 已经是物体中心高度。
        TCP 朝向与抓取时相同（竖直朝下）。
        放置时 TCP z = slot[2]（物体中心高度 = 桌面 + 物体半高）
        """
        R = np.array([
            [ 1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1],
        ], dtype=np.float64)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = slot   # [x, y, z_center]
        return T

    # ── 调试信息 ──────────────────────────────────────────────────────────

    def describe_plan(self, seq: ActionSequence) -> str:
        """返回规划结果的可读描述（用于 dry-run 测试）"""
        lines = [f"ActionSequence（共 {seq.n_actions} 步）："]
        ta = seq.scene_repr.target_area
        for i, act in enumerate(seq.actions):
            gp = act.grasp_position.round(3)
            pp = act.place_position.round(3)
            in_ta = ta.contains(pp[:2])
            lines.append(
                f"  [{i+1}] {act.object_id}\n"
                f"        grasp  → ({gp[0]}, {gp[1]}, {gp[2]})\n"
                f"        place  → ({pp[0]}, {pp[1]}, {pp[2]})  "
                f"{'✅ 在目标区' if in_ta else '❌ 不在目标区'}"
            )
        return "\n".join(lines)