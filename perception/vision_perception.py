# perception/vision_perception.py
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from mani_skill.utils.structs import Actor, Link

from perception.base import (
    BasePerception,
    ObjectInfo,
    TargetArea,
    SceneRepresentation,
)

_SPHERE_AXIS_DIFF = 0.025   # 三轴差值 < 1.5cm 判定为球形 
class VisionPerception(BasePerception):
    """
    基于视觉观测的感知模块（Task 3）。

    支持两种 obs_mode：
      - "sensor_data"           : 读取 PositionSegmentation (H,W,4) torch.int16
                                  前3通道为 XYZ (mm, OpenGL)，第4通道为 seg ID
      - "rgb+depth+segmentation": 读取独立的 depth (H,W,1) + segmentation (H,W,1)
                                  depth 单位 mm, torch.int16/uint16，0 为无效像素

    外参/内参统一从 obs["sensor_param"] 读取（OpenCV 约定），
    不再手动构建 intrinsics，也不使用 get_model_matrix()。

    设计原则：
      - 完全不读取 env 内部任何物理状态（seg 映射构建除外，仅在 reset 时执行一次）
      - rotation 设为 Identity（top-down 抓取场景下足够）
      - target_area 从 target_region actor 的点云估计
    """

    # 目标 actor 名称，与 multi_object_env 里的命名保持一致
    _TARGET_ACTOR_NAME = "target_region"

    def __init__(self, env, camera_name: str = "base_camera"):
        self._env          = env
        self._camera_name  = camera_name

        # seg_id → 描述 的映射，reset() 或首次 observe() 时构建。
        # 条目格式：
        #   object entry : {"type": "object", "object_id": str, "category": str}
        #   target entry : {"type": "target"}
        self._seg_map: Dict[int, dict] = {}

        # 检测当前 obs_mode，决定数据解析路径
        # 通过检查 env 是否有 obs_mode 属性来判断
        self._obs_mode: str = getattr(env.unwrapped, "obs_mode", "sensor_data")

    # ─────────────────────────────────────────────────────────────────────
    #  seg_id 映射（使用官方 segmentation_id_map，不依赖顺序）
    # ─────────────────────────────────────────────────────────────────────

    def _build_seg_map(self) -> None:
        """
        通过 env.segmentation_id_map 建立 per_scene_id → 元信息 的映射。

        官方推荐写法：遍历 segmentation_id_map，根据 Actor.name 判断角色。
        这比直接取 actor._objs[0].per_scene_id 更健壮，
        不依赖 env.objects 和 scene_cfg.objects 的顺序一致性。
        """
        self._seg_map = {}
        env = self._env.unwrapped

        # 预建立 object_id → category 的快查表
        id_to_category: Dict[str, str] = {
            cfg.id: cfg.category
            for cfg in env.scene_cfg.objects
        }

        for seg_id, actor_or_link in sorted(env.segmentation_id_map.items()):
            # 只处理 Actor 类型（Link 是机器人关节，不需要）
            if not isinstance(actor_or_link, Actor):
                continue

            name = actor_or_link.name

            if name == self._TARGET_ACTOR_NAME:
                self._seg_map[int(seg_id)] = {"type": "target"}

            elif name in id_to_category:
                self._seg_map[int(seg_id)] = {
                    "type":      "object",
                    "object_id": name,
                    "category":  id_to_category[name],
                }
            # 其余 Actor（机器人底座、桌面等）直接忽略

    # ─────────────────────────────────────────────────────────────────────
    #  传感器参数（OpenCV 约定，直接从 obs 读取，最准确）
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_sensor_params(
        obs: dict,
        camera_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回：
          K    : (3, 3) float64，OpenCV 内参
          T_cw : (4, 4) float64，world-to-camera，OpenCV 约定
    
        extrinsic_cv 的实际 shape 是 (1, 3, 4)，即 [R|t] 形式的
        3×4 投影矩阵（OpenCV 标准外参），不是 4×4 齐次矩阵。
        需要手动补最后一行 [0,0,0,1] 才能取逆。
        """
        params = obs["sensor_param"][camera_name]
    
        # ── 内参 ──────────────────────────────────────────────────────────
        intrinsic = params["intrinsic_cv"]
        if hasattr(intrinsic, "cpu"):
            intrinsic = intrinsic.cpu().numpy()
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        while intrinsic.ndim > 2:          # (1,3,3) → (3,3)
            intrinsic = intrinsic[0]
        K = intrinsic                      # (3, 3)
    
        # ── 外参 ──────────────────────────────────────────────────────────
        extrinsic = params["extrinsic_cv"]
        if hasattr(extrinsic, "cpu"):
            extrinsic = extrinsic.cpu().numpy()
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        while extrinsic.ndim > 2:          # (1,3,4) → (3,4)
            extrinsic = extrinsic[0]
    
        # extrinsic_cv 是 [R|t]，shape (3,4)
        # 补最后一行 [0,0,0,1] 凑成 (4,4) 齐次矩阵
        if extrinsic.shape == (3, 4):
            bottom  = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            T_cw    = np.vstack([extrinsic, bottom])   # (4, 4)
        elif extrinsic.shape == (4, 4):
            T_cw    = extrinsic                        # 已经是 4×4，直接用
        else:
            raise ValueError(
                f"[_parse_sensor_params] extrinsic_cv 的 shape 不认识: {extrinsic.shape}"
            )
    
        return K, T_cw     # (3,3), (4,4)

    # ─────────────────────────────────────────────────────────────────────
    #  数据读取：兼容 sensor_data 和 rgb+depth+segmentation 两种 obs_mode
    # ─────────────────────────────────────────────────────────────────────

    def _parse_depth_and_seg(
        self,
        obs: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回：
          depth : (H, W) float32，单位米，无效像素为 0.0
          seg   : (H, W) int32，  seg ID

        支持两种 obs_mode 的数据格式：
          1. sensor_data (minimal shader)：
               PositionSegmentation [H,W,4] int16
               前3通道 = XYZ in mm (OpenGL convention)
               第4通道 = seg ID

          2. rgb+depth+segmentation (后处理 shader)：
               depth        [H,W,1] int16/uint16，mm，0=无效
               segmentation [H,W,1] int16/uint16，seg ID
        """
        cam_data = obs["sensor_data"][self._camera_name]

        # ── 模式 1: sensor_data / minimal shader ──────────────────────────
        if "PositionSegmentation" in cam_data:
            raw = cam_data["PositionSegmentation"]
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            raw = np.array(raw, dtype=np.int32)
            if raw.ndim == 4:            # (1, H, W, 4) → (H, W, 4)
                raw = raw[0]

            # XYZ in mm，OpenGL 约定：Z 为负（朝屏幕外），取绝对值得深度
            # OpenGL: X right, Y up, Z out-of-screen（朝向观察者）
            # depth = -Z（相机前方 Z 为负）
            z_mm    = raw[..., 2].astype(np.float32)    # 可能为负
            depth   = np.abs(z_mm) / 1000.0             # 取绝对值并转米
            # 原始 Z=0 表示无效（背景），保持 0
            depth[z_mm == 0] = 0.0

            seg = raw[..., 3].astype(np.int32)
            return depth, seg

        # ── 模式 2: rgb+depth+segmentation ────────────────────────────────
        elif "depth" in cam_data and "segmentation" in cam_data:
            depth_raw = cam_data["depth"]
            seg_raw   = cam_data["segmentation"]

            if hasattr(depth_raw, "cpu"):
                depth_raw = depth_raw.cpu().numpy()
            if hasattr(seg_raw, "cpu"):
                seg_raw = seg_raw.cpu().numpy()

            depth_raw = np.array(depth_raw)
            seg_raw   = np.array(seg_raw)

            # 去掉 batch dim 和 channel dim
            if depth_raw.ndim == 4:      # (1, H, W, 1) → (H, W)
                depth_raw = depth_raw[0, ..., 0]
            elif depth_raw.ndim == 3:    # (H, W, 1) → (H, W)
                depth_raw = depth_raw[..., 0]

            if seg_raw.ndim == 4:
                seg_raw = seg_raw[0, ..., 0]
            elif seg_raw.ndim == 3:
                seg_raw = seg_raw[..., 0]

            # depth: uint16/int16 mm → float32 m，0 = 无效保持 0
            depth = depth_raw.astype(np.float32) / 1000.0
            depth[depth_raw == 0] = 0.0

            seg = seg_raw.astype(np.int32)
            return depth, seg

        else:
            raise KeyError(
                f"[VisionPerception] camera '{self._camera_name}' 的 sensor_data 中"
                f"既没有 'PositionSegmentation' 也没有 'depth'+'segmentation'。"
                f"实际 keys: {list(cam_data.keys())}"
            )

    # ─────────────────────────────────────────────────────────────────────
    #  点云反投影（OpenCV 约定，与 intrinsic_cv / extrinsic_cv 对应）
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _depth_to_pointcloud_world(
        depth_raw: np.ndarray,   # (H, W) int16 或 float，原始值（毫米）
        K:         np.ndarray,   # (3, 3)
        T_cw:      np.ndarray,   # (4, 4)
    ) -> np.ndarray:             # (N, 3) 世界坐标，单位米
        """
        depth 原始值单位是毫米（int16），先转成米再反投影。
        """
        # ── 毫米 → 米 ──────────────────────────────────────────────────
        depth_m = depth_raw.astype(np.float64)   # (H, W)
    
        H, W = depth_m.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    
        u = np.arange(W, dtype=np.float64)
        v = np.arange(H, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
    
        # 像素 → 相机坐标系（OpenCV：Z 朝前）
        z = depth_m
        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z
    
        pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (H*W, 3)
    
        # 相机坐标 → 世界坐标
        T_wc  = np.linalg.inv(T_cw)
        ones  = np.ones((pts_cam.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts_cam, ones])                     # (H*W, 4)
        pts_world = (T_wc @ pts_h.T).T[:, :3]                 # (H*W, 3)
    
        return pts_world

    # ─────────────────────────────────────────────────────────────────────
    #  几何估计（点云 → 中心 / 尺寸 / pose）
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_sphere_lstsq(pts: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        代数法最小二乘球面拟合。
        求解 (x-cx)^2 + (y-cy)^2 + (z-cz)^2 = r^2
        线性化为：2cx*x + 2cy*y + 2cz*z + (r^2 - cx^2 - cy^2 - cz^2) = x^2+y^2+z^2
        """
        A = np.hstack([2 * pts, np.ones((len(pts), 1))])   # (N, 4)
        b = (pts ** 2).sum(axis=1)                          # (N,)
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, cz = result[:3]
        r = np.sqrt(result[3] + cx**2 + cy**2 + cz**2)
        return np.array([cx, cy, cz], dtype=np.float32), float(r)
        
    @staticmethod
    def _estimate_geometry(
        pts:     np.ndarray,
        min_pts: int = 10,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if len(pts) < min_pts:
            return None
    
        center     = pts.mean(axis=0).astype(np.float32)
        pt_min     = pts.min(axis=0)
        pt_max     = pts.max(axis=0)
        dimensions = (pt_max - pt_min).astype(np.float32)
        dimensions = np.maximum(dimensions, 1e-3)
    
        # ── 球形检测与修正 ─────────────────────────────────────────────
        axis_diff = dimensions.max() - dimensions.min()
        if axis_diff < _SPHERE_AXIS_DIFF and len(pts) >= 50:
            center_fit, radius = VisionPerception._fit_sphere_lstsq(pts)
            diameter   = 2.0 * radius
            dimensions = np.array([diameter, diameter, diameter], dtype=np.float32)
            center     = center_fit
    
        pose        = np.eye(4, dtype=np.float32)
        pose[:3, 3] = center
        return center, dimensions, pose

    # ─────────────────────────────────────────────────────────────────────
    #  目标区域估计
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_target_area(
        pts: np.ndarray,         # target_region 的世界系点云，已去 nan
    ) -> Optional[TargetArea]:
        """
        center = XY 质心
        size   = XY 方向 AABB 尺寸
        table_z = Z 最小值（薄板贴桌面）
        至少需要 5 个有效点。
        """
        if len(pts) < 5:
            return None

        center_xy = pts[:, :2].mean(axis=0).astype(np.float32)
        xy_min    = pts[:, :2].min(axis=0)
        xy_max    = pts[:, :2].max(axis=0)
        size_xy   = (xy_max - xy_min).astype(np.float32)
        table_z   = float(pts[:, 2].min())

        return TargetArea(
            center  = center_xy,
            size    = size_xy,
            table_z = table_z,
        )

    # ─────────────────────────────────────────────────────────────────────
    #  主接口
    # ─────────────────────────────────────────────────────────────────────

    def observe(self, obs: dict) -> SceneRepresentation:
        # 0. 首次调用时建立 seg 映射（或 reset 后重建）
        if not self._seg_map:
            self._build_seg_map()

        # 1. 解析深度图和分割图（自动兼容两种 obs_mode）
        depth, seg = self._parse_depth_and_seg(obs)

        # 2. 从 obs["sensor_param"] 读取内外参（OpenCV 约定，最准确）
        K, T_cw = self._parse_sensor_params(obs, self._camera_name)

        # 3. 整图反投影为世界系点云
        pts_world = self._depth_to_pointcloud_world(depth, K, T_cw)  # (H*W, 3)
        seg_flat  = seg.reshape(-1)

        # 4. 按 seg_id 分组处理
        objects:     list[ObjectInfo]      = []
        target_area: Optional[TargetArea]  = None

        for seg_id, meta in self._seg_map.items():
            mask    = seg_flat == seg_id
            pts_obj = pts_world[mask]
            valid   = ~np.isnan(pts_obj).any(axis=1)
            pts_obj = pts_obj[valid]

            if meta["type"] == "target":
                target_area = self._estimate_target_area(pts_obj)
                if target_area is None:
                    print(f"[VisionPerception] WARN: target_region 点云不足 "
                          f"({len(pts_obj)} pts)，无法估计目标区域。")

            elif meta["type"] == "object":
                result = self._estimate_geometry(pts_obj)
                if result is None:
                    print(f"[VisionPerception] WARN: {meta['object_id']} "
                          f"点云不足（{len(pts_obj)} pts），跳过。")
                    continue
                _, dimensions, pose = result
                objects.append(ObjectInfo(
                    object_id  = meta["object_id"],
                    category   = meta["category"],
                    pose       = pose,
                    dimensions = dimensions,
                    confidence = float(np.clip(len(pts_obj) / 300.0, 0.0, 1.0)),
                ))

        # 5. target_area 估计失败时的 fallback
        if target_area is None:
            print("[VisionPerception] WARN: 使用零值 TargetArea 作为 fallback。")
            target_area = TargetArea(
                center  = np.zeros(2, dtype=np.float32),
                size    = np.zeros(2, dtype=np.float32),
                table_z = 0.0,
            )

        return SceneRepresentation(
            objects         = objects,
            target_area     = target_area,
            perception_mode = "vision",
        )

    def reset(self) -> None:
        """episode 开始时清空 seg 映射，下次 observe 时重建。"""
        self._seg_map = {}