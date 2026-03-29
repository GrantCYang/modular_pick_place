# envs/scene_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ─────────────────────────────────────────────
#  GraspMetadataEntry：yaml 里单条抓取记录
# ─────────────────────────────────────────────

@dataclass
class GraspMetadataEntry:
    """
    物体局部坐标系下的抓取描述。
    local_position: EEF 目标位置相对于物体中心的偏移，单位：米
    local_rpy:      EEF 姿态的 roll-pitch-yaw，相对于物体局部坐标系，单位：弧度
    width:          建议夹爪开合宽度，单位：米
    score:          抓取质量分
    """
    local_position: np.ndarray    # shape (3,)
    local_rpy:      np.ndarray    # shape (3,)，roll-pitch-yaw
    width:          float
    score:          float


# ─────────────────────────────────────────────
#  ObjectConfig
# ─────────────────────────────────────────────

@dataclass
class ObjectConfig:
    id:       str
    category: str                       # "box" | "cylinder" | "ycb"
    color:    Optional[List[float]]     # RGBA；ycb 可为 None

    # box 专用
    half_size: Optional[List[float]] = None

    # cylinder 专用
    radius:      Optional[float] = None
    half_length: Optional[float] = None

    # ycb 专用
    model_id:          Optional[str]   = None
    z_offset_override: Optional[float] = None

    # ycb 抓取 metadata（局部坐标系）
    grasp_metadata: List[GraspMetadataEntry] = field(default_factory=list)

    # _after_reconfigure 写入的运行时缓存，不来自 yaml
    _dimensions_cache: Optional[np.ndarray] = field(default=None, repr=False)

    def validate(self):
        if self.category == "box":
            assert self.half_size is not None, \
                f"[{self.id}] box 必须提供 half_size"
        elif self.category == "cylinder":
            assert self.radius is not None and self.half_length is not None, \
                f"[{self.id}] cylinder 必须提供 radius 和 half_length"
        elif self.category == "ycb":
            assert self.model_id is not None, \
                f"[{self.id}] ycb 必须提供 model_id"
        else:
            raise ValueError(
                f"[{self.id}] 未知 category: {self.category}，"
                f"支持 box / cylinder / ycb"
            )

    @property
    def z_offset(self) -> float:
        if self.z_offset_override is not None:
            return self.z_offset_override
        if self.category == "box":
            return self.half_size[2]
        elif self.category == "cylinder":
            return self.half_length
        elif self.category == "ycb":
            return 0.05
        return 0.05

    @property
    def dimensions(self) -> np.ndarray:
        if self._dimensions_cache is not None:
            return self._dimensions_cache
        if self.category == "box":
            return np.array(self.half_size) * 2
        elif self.category == "cylinder":
            return np.array([
                self.radius * 2,
                self.radius * 2,
                self.half_length * 2,
            ])
        elif self.category == "ycb":
            return np.array([0.1, 0.1, 0.1])    # 占位，_after_reconfigure 会覆盖
        return np.array([0.05, 0.05, 0.05])


# ─────────────────────────────────────────────
#  TargetAreaConfig / SpawnConfig / SceneConfig
# ─────────────────────────────────────────────

@dataclass
class TargetAreaConfig:
    center: np.ndarray
    size:   np.ndarray
    color:  List[float]

    @property
    def half(self) -> np.ndarray:
        return self.size / 2


@dataclass
class SpawnConfig:
    x_range:      List[float]
    y_range:      List[float]
    min_distance: float


@dataclass
class SceneConfig:
    objects:     List[ObjectConfig]
    target_area: TargetAreaConfig
    spawn:       SpawnConfig

    def validate(self):
        assert len(self.objects) > 0, "至少需要 1 个物体"
        for obj in self.objects:
            obj.validate()
        ids = [o.id for o in self.objects]
        assert len(ids) == len(set(ids)), f"物体 id 有重复: {ids}"


# ─────────────────────────────────────────────
#  YAML 加载
# ─────────────────────────────────────────────

def _parse_grasp_metadata(raw_list: list) -> List[GraspMetadataEntry]:
    entries = []
    for item in raw_list:
        entries.append(GraspMetadataEntry(
            local_position=np.array(item["local_position"], dtype=np.float32),
            local_rpy=np.array(item["local_rpy"],      dtype=np.float32),
            width=float(item["width"]),
            score=float(item["score"]),
        ))
    return entries


def load_scene_config(path) -> SceneConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    objects = []
    for item in raw["objects"]:
        # 解析 grasp_metadata（仅 ycb 需要，其余为空列表）
        gm_raw = item.get("grasp_metadata", {})
        grasp_metadata = _parse_grasp_metadata(gm_raw.get("candidates", []))

        obj = ObjectConfig(
            id=item["id"],
            category=item["category"],
            color=item.get("color"),
            half_size=item.get("half_size"),
            radius=item.get("radius"),
            half_length=item.get("half_length"),
            model_id=item.get("model_id"),
            z_offset_override=item.get("z_offset_override"),
            grasp_metadata=grasp_metadata,
        )
        obj.validate()
        objects.append(obj)

    ta = raw["target_area"]
    target_area = TargetAreaConfig(
        center=np.array(ta["center"], dtype=np.float32),
        size=np.array(ta["size"],     dtype=np.float32),
        color=ta["color"],
    )

    sp = raw["spawn"]
    spawn = SpawnConfig(
        x_range=sp["x_range"],
        y_range=sp["y_range"],
        min_distance=sp["min_distance"],
    )

    cfg = SceneConfig(objects=objects, target_area=target_area, spawn=spawn)
    cfg.validate()
    return cfg