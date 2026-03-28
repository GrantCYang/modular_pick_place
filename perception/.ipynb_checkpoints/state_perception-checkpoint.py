# perception/state_perception.py
from __future__ import annotations
import numpy as np
from perception.base import (
    BasePerception,
    ObjectInfo,
    TargetArea,
    SceneRepresentation,
)


class StatePerception(BasePerception):
    """
    基于特权状态的感知模块。
    直接读取 env.get_privileged_state()，将其转换为 SceneRepresentation。
    不做任何估计或近似，所有值都是 ground-truth。
    """

    def __init__(self, env):
        # 只持有 env 引用，Planning/Execution 永远拿不到这个 env
        self._env = env

    def observe(self, obs: dict) -> SceneRepresentation:
        """
        obs 参数在这里实际上不使用（特权状态直接从 env 读取）。
        保留 obs 参数是为了遵守 BasePerception 接口契约，
        确保 StatePerception 和 VisionPerception 可以无缝替换。
        """
        raw = self._env.unwrapped.get_privileged_state()

        # ── 构建 ObjectInfo 列表 ──────────────────────────────────────────
        objects = []
        for obj_raw in raw["objects"]:
            objects.append(ObjectInfo(
                object_id  = obj_raw["id"],
                category   = obj_raw["category"],
                pose       = obj_raw["pose"],        # 已经是 4×4 np.ndarray
                dimensions = obj_raw["dimensions"],  # 已经是 (3,) np.ndarray
                confidence = 1.0,                    # 特权状态，置信度恒为 1
            ))

        # ── 构建 TargetArea ───────────────────────────────────────────────
        t = raw["target"]
        target_area = TargetArea(
            center  = t["center"],   # (2,) np.ndarray
            size    = t["size"],     # (2,) np.ndarray
            table_z = t["table_z"],  # float
        )

        return SceneRepresentation(
            objects         = objects,
            target_area     = target_area,
            perception_mode = "state",
        )