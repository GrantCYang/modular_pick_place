# envs/multi_object_env.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.robots import Panda, Fetch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .scene_config import SceneConfig, load_scene_config

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "scene.yaml"


@register_env("MultiObjectPickAndPlace-v1", max_episode_steps=500)
class MultiObjectPickAndPlaceEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Panda

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        scene_config: SceneConfig | str | Path | None = None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        if isinstance(scene_config, SceneConfig):
            self.scene_cfg = scene_config
        elif scene_config is not None:
            self.scene_cfg = load_scene_config(scene_config)
        else:
            self.scene_cfg = load_scene_config(_DEFAULT_CONFIG)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ── 相机配置 ──────────────────────────────────────────────────────────

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye    = [0.0, -0.2, 0.5],   # 物体区域侧后方
            target = [-0.22, 0.0, 0.0],  # 对准物体生成区域中心
        )
        return [CameraConfig(
            "base_camera",
            pose,
            128, 128,
            np.pi / 3,  # 60° FOV
            0.01, 100,
        )]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.7, 0.6], target=[0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # ── 机器人加载 ────────────────────────────────────────────────────────

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # ── 场景构建 ──────────────────────────────────────────────────────────

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.objects: List[sapien.Entity] = []
        for obj_cfg in self.scene_cfg.objects:
            actor = self._build_object(obj_cfg)
            self.objects.append(actor)

        ta = self.scene_cfg.target_area
        self.target_region = self._build_thin_box(
            half_size=[ta.half[0], ta.half[1], 0.001],
            color=ta.color,
            name="target_region",
        )

    def _build_object(self, obj_cfg) -> sapien.Entity:
        """根据 ObjectConfig.category 分发到对应的 builder"""
        if obj_cfg.category == "box":
            return self._build_box(obj_cfg)
        elif obj_cfg.category == "cylinder":
            return self._build_cylinder(obj_cfg)
        elif obj_cfg.category == "ycb":                  # ← 新增分支
            return self._build_ycb(obj_cfg)
        else:
            raise ValueError(f"不支持的 category: {obj_cfg.category}")

    def _build_box(self, obj_cfg) -> sapien.Entity:
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        builder.add_box_collision(half_size=obj_cfg.half_size)
        builder.add_box_visual(
            half_size=obj_cfg.half_size,
            material=sapien.render.RenderMaterial(base_color=obj_cfg.color),
        )
        return builder.build(name=obj_cfg.id)

    def _build_cylinder(self, obj_cfg) -> sapien.Entity:
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        builder.add_cylinder_collision(
            radius=obj_cfg.radius, half_length=obj_cfg.half_length
        )
        builder.add_cylinder_visual(
            radius=obj_cfg.radius,
            half_length=obj_cfg.half_length,
            material=sapien.render.RenderMaterial(base_color=obj_cfg.color),
        )
        return builder.build(name=obj_cfg.id)

    def _build_ycb(self, obj_cfg) -> sapien.Entity:
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{obj_cfg.model_id}",
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])  # 与官方对齐，不再用 0.5
        return builder.build(name=obj_cfg.id)

    def _build_thin_box(self, half_size, color, name) -> sapien.Entity:
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[0, 0, -1])
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(
            half_size=half_size,
            material=sapien.render.RenderMaterial(base_color=color),
        )
        return builder.build_static(name=name)

    # ── YCB z_offset 精确计算 ─────────────────────────────────────────────

    def _after_reconfigure(self, options: dict):
        for actor, obj_cfg in zip(self.objects, self.scene_cfg.objects):
            if obj_cfg.category != "ycb":
                continue
            try:
                mesh   = actor.get_first_collision_mesh()  # to_world_frame=True，initial_pose=[0,0,0]
                bounds = mesh.bounding_box.bounds          # (2, 3)，此时等价于局部坐标系
    
                # 真实尺寸
                obj_cfg._dimensions_cache = (
                    bounds[1] - bounds[0]
                ).astype(np.float32)
    
                # 质心偏移：局部坐标系下几何中心相对于 mesh 原点的偏移
                obj_cfg._centroid_offset_cache = (
                    (bounds[0] + bounds[1]) / 2
                ).astype(np.float32)
    
                # 自动 z_offset：底部贴桌面，仅在没有手动 override 时计算
                if obj_cfg.z_offset_override is None:
                    obj_cfg.z_offset_override = float(-bounds[0, 2])
    
                print(f"[_after_reconfigure] {obj_cfg.id}: "
                      f"dims={obj_cfg._dimensions_cache.round(4)}, "
                      f"centroid_offset={obj_cfg._centroid_offset_cache.round(4)}, "
                      f"z_offset={obj_cfg.z_offset_override:.4f}")
    
            except Exception as e:
                import traceback
                traceback.print_exc()

    # ── Episode 初始化 ────────────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            table_z = self._get_table_surface_z()
            ta = self.scene_cfg.target_area
            sp = self.scene_cfg.spawn

            # ① 固定目标区域位置
            target_p = torch.zeros((b, 3), device=self.device)
            target_p[:, 0] = float(ta.center[0])
            target_p[:, 1] = float(ta.center[1])
            target_p[:, 2] = table_z + 0.001
            self.target_region.set_pose(
                Pose.create_from_pq(p=target_p, q=[1, 0, 0, 0])
            )

            # ② 随机放置每个物体
            placed_xy: List[np.ndarray] = []
            for actor, obj_cfg in zip(self.objects, self.scene_cfg.objects):
                pos_xy = self._sample_valid_position(placed_xy, sp, ta)
                placed_xy.append(pos_xy)

                obj_p = torch.zeros((b, 3), device=self.device)
                obj_p[:, 0] = float(pos_xy[0])
                obj_p[:, 1] = float(pos_xy[1])
                obj_p[:, 2] = table_z + obj_cfg.z_offset
                actor.set_pose(Pose.create_from_pq(p=obj_p, q=[1, 0, 0, 0]))

    # ── 工具方法 ──────────────────────────────────────────────────────────

    def _get_table_surface_z(self) -> float:
        return 0.0

    def _sample_valid_position(
        self,
        existing: List[np.ndarray],
        sp,
        ta,
        max_attempts: int = 200,
    ) -> np.ndarray:
        for _ in range(max_attempts):
            x = np.random.uniform(*sp.x_range)
            y = np.random.uniform(*sp.y_range)
            pos = np.array([x, y])

            if np.all(np.abs(pos - ta.center) <= ta.half + 0.03):
                continue

            if any(np.linalg.norm(pos - ep) < sp.min_distance for ep in existing):
                continue

            return pos

        raise RuntimeError(
            f"采样失败（尝试 {max_attempts} 次）。"
            "请检查 YAML 中的 spawn 范围或 min_distance 设置。"
        )

    def _batch_in_target_area(self, xy_batch: torch.Tensor) -> torch.Tensor:
        ta = self.scene_cfg.target_area
        center = torch.tensor(ta.center, device=self.device, dtype=torch.float32)
        half   = torch.tensor(ta.half,   device=self.device, dtype=torch.float32)
        return torch.all(torch.abs(xy_batch - center) <= half, dim=-1)

    # ── 奖励 ──────────────────────────────────────────────────────────────

    def evaluate(self):
        success = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for actor in self.objects:
            in_target = self._batch_in_target_area(actor.pose.p[:, :2])
            success &= in_target
        return {"success": success}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = torch.zeros(self.num_envs, device=self.device)
        for actor in self.objects:
            in_target = self._batch_in_target_area(actor.pose.p[:, :2])
            reward += in_target.float()
        return reward / max(len(self.objects), 1)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)

    # ── 特权状态接口 ──────────────────────────────────────────────────────

    def get_privileged_state(self) -> Dict[str, Any]:
        table_z = self._get_table_surface_z()
        objects_state = []
    
        for actor, obj_cfg in zip(self.objects, self.scene_cfg.objects):
            p = actor.pose.p[0].cpu().numpy()          # mesh 原点在世界坐标系的位置
            q = actor.pose.q[0].cpu().numpy()          # wxyz
    
            rot = Rotation.from_quat(
                [q[1], q[2], q[3], q[0]]
            ).as_matrix()
    
            # ── 质心修正：mesh 原点 → 几何质心 ──────────────────────────────
            offset = getattr(obj_cfg, "_centroid_offset_cache", None)
            if offset is not None:
                # offset 是局部坐标系下的偏移，需要旋转到世界坐标系
                p = p + rot @ offset.astype(np.float64)
    
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3]  = p
    
            dims = getattr(obj_cfg, "_dimensions_cache", None)
            if dims is None:
                dims = obj_cfg.dimensions
    
            objects_state.append({
                "id":         obj_cfg.id,
                "category":   obj_cfg.category,
                "pose":       T,
                "dimensions": dims,
            })
    
        ta = self.scene_cfg.target_area
        return {
            "objects": objects_state,
            "target": {
                "center":  ta.center.copy(),
                "size":    ta.size.copy(),
                "table_z": table_z,
            },
        }

    def get_success_info(self) -> Dict[str, bool]:
        ta = self.scene_cfg.target_area
        result = {}
        for actor, obj_cfg in zip(self.objects, self.scene_cfg.objects):
            pos_xy = actor.pose.p[0, :2].cpu().numpy()
            in_area = bool(np.all(np.abs(pos_xy - ta.center) <= ta.half + 0.01))
            result[obj_cfg.id] = in_area
        return result