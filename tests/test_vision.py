# tests/test_vision.py
"""
专门诊断 VisionPerception 点云为 0 的问题。
一次性打印所有关键中间量：
  1. segmentation_id_map 里有什么 ID
  2. 深度图的值域（判断深度是否有效）
  3. 分割图里实际出现了哪些 ID
  4. seg_map 建立了什么映射
  5. 每个 seg_id 在分割图里匹配到了几个像素
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import mani_skill.envs                                         # noqa
from mani_skill.utils.structs import Actor, Link
from envs.multi_object_env import MultiObjectPickAndPlaceEnv   # noqa


def _to_np(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _strip(x, ndim):
    while x.ndim > ndim:
        x = x[0]
    return x


env = gym.make(
    "MultiObjectPickAndPlace-v1",
    render_mode  = "rgb_array",
    obs_mode     = "rgb+depth+segmentation",
    scene_config = str(ROOT / "config" / "scene.yaml"),
)
obs, _ = env.reset(seed=0)
unwrapped = env.unwrapped

# ══════════════════════════════════════════════════════════════
# 1. segmentation_id_map 里有什么
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("1. segmentation_id_map 内容")
print("═"*60)
seg_id_map = unwrapped.segmentation_id_map
for sid, obj in sorted(seg_id_map.items()):
    kind = type(obj).__name__
    name = getattr(obj, "name", "???")
    print(f"  seg_id={sid:4d}  type={kind:<6}  name={name}")

# ══════════════════════════════════════════════════════════════
# 2. 深度图值域
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("2. 深度图 (depth) 原始值域")
print("═"*60)
cam_data = obs["sensor_data"]["base_camera"]
print(f"  sensor_data keys: {list(cam_data.keys())}")

depth_raw = _strip(_to_np(cam_data["depth"]), 3)[..., 0]
print(f"  depth shape={depth_raw.shape}  dtype={depth_raw.dtype}")
print(f"  min={depth_raw.min()}  max={depth_raw.max()}")
print(f"  nonzero pixels: {(depth_raw != 0).sum()} / {depth_raw.size}")
print(f"  unique values (前20个): {np.unique(depth_raw)[:20]}")

# ══════════════════════════════════════════════════════════════
# 3. 分割图里实际出现的 ID
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("3. 分割图 (segmentation) 实际出现的 ID")
print("═"*60)
seg_raw = _strip(_to_np(cam_data["segmentation"]), 3)[..., 0].astype(np.int32)
print(f"  seg shape={seg_raw.shape}  dtype={seg_raw.dtype}")
unique_ids, counts = np.unique(seg_raw, return_counts=True)
for uid, cnt in zip(unique_ids, counts):
    name_in_map = seg_id_map.get(uid, "NOT IN MAP")
    if hasattr(name_in_map, "name"):
        name_in_map = name_in_map.name
    print(f"  seg_id={uid:4d}  pixels={cnt:6d}  → {name_in_map}")

# ══════════════════════════════════════════════════════════════
# 4. scene_cfg.objects 里有什么 ID
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("4. scene_cfg.objects")
print("═"*60)
for cfg in unwrapped.scene_cfg.objects:
    print(f"  id={cfg.id}  category={cfg.category}")

# ══════════════════════════════════════════════════════════════
# 5. env.objects 里 Actor 的 name 和 per_scene_id
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("5. env.objects Actor 信息")
print("═"*60)
for actor in unwrapped.objects:
    name = actor.name
    # per_scene_id 可能在 _objs[0] 里
    try:
        psid = actor._objs[0].per_scene_id
    except Exception:
        psid = "N/A"
    print(f"  name={name}  per_scene_id={psid}")

# ══════════════════════════════════════════════════════════════
# 6. _build_seg_map 会建立什么映射（手动模拟）
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("6. 模拟 _build_seg_map 的结果")
print("═"*60)
id_to_category = {cfg.id: cfg.category for cfg in unwrapped.scene_cfg.objects}
TARGET_NAME = "target_region"

built_map = {}
for seg_id, actor_or_link in sorted(seg_id_map.items()):
    if not isinstance(actor_or_link, Actor):
        continue
    name = actor_or_link.name
    if name == TARGET_NAME:
        built_map[int(seg_id)] = {"type": "target"}
    elif name in id_to_category:
        built_map[int(seg_id)] = {"type": "object", "object_id": name,
                                  "category": id_to_category[name]}

print(f"  built_map:")
for k, v in built_map.items():
    print(f"    seg_id={k}  →  {v}")

# 关键检查：built_map 里的 seg_id 是否出现在分割图里
print(f"\n  交叉验证（built_map seg_id 是否在分割图中出现）:")
for seg_id, meta in built_map.items():
    pixels = (seg_raw == seg_id).sum()
    print(f"    seg_id={seg_id}  ({meta})  → 分割图中像素数={pixels}")

# ══════════════════════════════════════════════════════════════
# 7. 相机位置（判断是否能看到场景）
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("7. 相机位姿（cam2world_gl，世界坐标系位置）")
print("═"*60)
cam_param = obs["sensor_param"]["base_camera"]
c2w = _strip(_to_np(cam_param["cam2world_gl"]), 2)
ext = _strip(_to_np(cam_param["extrinsic_cv"]), 2)
intr = _strip(_to_np(cam_param["intrinsic_cv"]), 2)
print(f"  cam2world_gl (OpenGL):\n{c2w}")
print(f"  extrinsic_cv [R|t] (3×4):\n{ext}")
print(f"  intrinsic_cv:\n{intr}")
cam_pos = c2w[:3, 3]
print(f"\n  相机世界坐标: x={cam_pos[0]:.4f}  y={cam_pos[1]:.4f}  z={cam_pos[2]:.4f}")
print(f"  物体世界坐标（来自 state）:")
for actor in unwrapped.objects:
    try:
        p = actor.pose.p
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        p = np.asarray(p).flatten()
        print(f"    {actor.name}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
    except Exception as e:
        print(f"    {actor.name}: 读取失败 {e}")

env.close()
print("\n" + "═"*60)
print("诊断完成。请将以上输出完整贴出。")
print("═"*60)


# tests/debug_vision.py 末尾追加以下内容（env.close() 之前）
# ══════════════════════════════════════════════════════════════
# 8. 直接可视化相机拍到的内容
# ══════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("base_camera 实际拍到的内容", fontsize=13)

# ── RGB ──────────────────────────────────────────────────────
ax = axes[0]
ax.set_title("RGB")
ax.axis("off")
rgb = _strip(_to_np(cam_data["rgb"]), 3)[..., :3]
ax.imshow(rgb.astype(np.uint8))

# ── Depth（毫米，伪彩）────────────────────────────────────────
ax = axes[1]
ax.set_title(f"Depth (int16, mm)\nmin={depth_raw.min()} max={depth_raw.max()}")
ax.axis("off")
im = ax.imshow(depth_raw, cmap="plasma")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("mm")

# ── Segmentation（物体高亮）──────────────────────────────────
ax = axes[2]
ax.set_title("Segmentation\n(物体区域高亮)")
ax.axis("off")

cmap20 = cm.get_cmap("tab20", 20)
seg_vis = np.ones((*seg_raw.shape, 3), dtype=np.float32) * 0.85  # 灰色背景

object_seg_ids = {18: "box_0", 19: "cylinder_0", 20: "mustard_0", 21: "target"}
colors_used = {}
for idx, (sid, name) in enumerate(object_seg_ids.items()):
    color = np.array(cmap20(idx)[:3])
    mask  = seg_raw == sid
    seg_vis[mask] = color
    colors_used[name] = color
    # 像素数标注
    n_px = mask.sum()
    if n_px > 0:
        ys, xs = np.where(mask)
        cx_px, cy_px = xs.mean(), ys.mean()
        ax.text(cx_px, cy_px, f"{name}\n{n_px}px",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))

ax.imshow(seg_vis)
patches = [mpatches.Patch(color=c, label=n) for n, c in colors_used.items()]
ax.legend(handles=patches, fontsize=7, loc="lower right")

plt.tight_layout()
out = "debug_camera_raw.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n📷 相机原始图像已保存：{out}")