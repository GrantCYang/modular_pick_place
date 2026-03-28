# envs/scene_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import yaml


# ── 单个物体的配置 ──────────────────────────────────────────────────────────

@dataclass
class ObjectConfig:
    id:       str
    category: str                          # "box" | "cylinder"
    color:    List[float]                  # RGBA

    # box 专用
    half_size: Optional[List[float]] = None

    # cylinder 专用
    radius:      Optional[float] = None
    half_length: Optional[float] = None

    def validate(self):
        if self.category == "box":
            assert self.half_size is not None, \
                f"[{self.id}] box 必须提供 half_size"
        elif self.category == "cylinder":
            assert self.radius is not None and self.half_length is not None, \
                f"[{self.id}] cylinder 必须提供 radius 和 half_length"
        else:
            raise ValueError(f"[{self.id}] 未知 category: {self.category}，"
                             f"目前支持 box / cylinder")

    @property
    def z_offset(self) -> float:
        """物体中心距桌面的高度偏移"""
        if self.category == "box":
            return self.half_size[2]
        elif self.category == "cylinder":
            return self.half_length
        return 0.05

    @property
    def dimensions(self) -> np.ndarray:
        """用于 get_privileged_state 的 bounding box 尺寸 (全长)"""
        if self.category == "box":
            return np.array(self.half_size) * 2
        elif self.category == "cylinder":
            return np.array([self.radius * 2, self.radius * 2, self.half_length * 2])
        return np.array([0.05, 0.05, 0.05])


# ── 目标区域配置 ────────────────────────────────────────────────────────────

@dataclass
class TargetAreaConfig:
    center: np.ndarray           # shape (2,)
    size:   np.ndarray           # shape (2,)  [width, height]
    color:  List[float]          # RGBA

    @property
    def half(self) -> np.ndarray:
        return self.size / 2


# ── 生成范围配置 ────────────────────────────────────────────────────────────

@dataclass
class SpawnConfig:
    x_range:      List[float]    # [min, max]
    y_range:      List[float]    # [min, max]
    min_distance: float


# ── 整体场景配置 ────────────────────────────────────────────────────────────

@dataclass
class SceneConfig:
    objects:     List[ObjectConfig]
    target_area: TargetAreaConfig
    spawn:       SpawnConfig

    def validate(self):
        assert len(self.objects) > 0, "至少需要 1 个物体"
        for obj in self.objects:
            obj.validate()
        # id 唯一性检查
        ids = [o.id for o in self.objects]
        assert len(ids) == len(set(ids)), f"物体 id 有重复: {ids}"


# ── YAML 加载函数 ───────────────────────────────────────────────────────────

def load_scene_config(path: str | Path) -> SceneConfig:
    """
    从 YAML 文件加载场景配置。

    用法：
        cfg = load_scene_config("config/scene.yaml")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # ── 解析 objects ──────────────────────────────────────────────────────
    objects = []
    for item in raw["objects"]:
        obj = ObjectConfig(
            id=item["id"],
            category=item["category"],
            color=item["color"],
            half_size=item.get("half_size"),
            radius=item.get("radius"),
            half_length=item.get("half_length"),
        )
        obj.validate()
        objects.append(obj)

    # ── 解析 target_area ──────────────────────────────────────────────────
    ta = raw["target_area"]
    target_area = TargetAreaConfig(
        center=np.array(ta["center"], dtype=np.float32),
        size=np.array(ta["size"],   dtype=np.float32),
        color=ta["color"],
    )

    # ── 解析 spawn ────────────────────────────────────────────────────────
    sp = raw["spawn"]
    spawn = SpawnConfig(
        x_range=sp["x_range"],
        y_range=sp["y_range"],
        min_distance=sp["min_distance"],
    )

    cfg = SceneConfig(objects=objects, target_area=target_area, spawn=spawn)
    cfg.validate()
    return cfg