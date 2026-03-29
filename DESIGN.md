# Design Document: Modular Pick and Place

## 1. Module Interfaces

The system is decomposed into three modules with strictly defined boundaries.
No module accesses the simulator directly except through its designated inputs.

### 1.1 Perception

| | |
|---|---|
| **Input** | `obs: dict` — raw observation from `env.reset()` / `env.step()` |
| **Output** | `SceneRepresentation` — structured scene description |

Output schema:

```python  
SceneRepresentation(  
    objects:          List[ObjectInfo],  # one entry per object in the scene  
    target_area:      TargetArea,        # placement target specification  
    perception_mode:  str,               # "state" or "vision"  
    timestamp:        float,             # current env step count  
)  

ObjectInfo(  
    object_id:   str,           # unique identifier, e.g. "box_0"  
    category:    str,           # "box", "ycb", etc.  
    pose:        np.ndarray,    # shape (4, 4), SE3 in world frame  
    dimensions:  np.ndarray,    # shape (3,), AABB size (x, y, z) in metres  
    confidence:  float,         # 1.0 for state-based; <1.0 for vision-based  
)  

TargetArea(  
    center:   np.ndarray,  # shape (2,), (x, y) in world frame  
    size:     np.ndarray,  # shape (2,), (width_x, width_y) in metres  
    table_z:  float,       # table surface height in metres  
)  
```

Two concrete implementations share this interface:

- **`StatePerception`** — reads ground-truth poses and sizes directly from
  simulator privileged state. Confidence is always 1.0.
- **`VisionPerception`** — processes RGB-D images and semantic segmentation
  masks from the onboard camera. Estimates object pose and dimensions from
  the point cloud of each segmented region. For spherical objects (e.g. the
  YCB orange), a dedicated sphere-fitting step replaces the raw centroid
  estimate to correct the systematic upward bias caused by partial occlusion
  against the table surface.

Both subclass `BasePerception` and implement a single method:

```python  
def observe(self, obs: dict) -> SceneRepresentation: ...  
```

---

### 1.2 Planning

| | |
|---|---|
| **Input** | `SceneRepresentation` |
| **Output** | `ActionSequence` |

Output schema:

```python  
ActionSequence(  
    actions:     List[GraspAction],   # one entry per object, ordered  
    scene_repr:  SceneRepresentation, # back-reference for debugging  
)  

GraspAction(  
    object_id:   str,          # which object this action targets  
    grasp_pose:  np.ndarray,   # shape (4, 4), TCP target pose for grasping  
    place_pose:  np.ndarray,   # shape (4, 4), TCP target pose for placing  
    lift_height: float,        # metres above grasp point during transport  
    object_dims: np.ndarray,   # shape (3,), copied from ObjectInfo  
)  
```

`SequentialPlanner` implements `BasePlanner.plan()` in three steps:

1. **Slot allocation** — divides the target area into a grid of
   non-overlapping slots, one per object. Grid spacing accounts for each
   object's footprint. X-axis slots are ordered far-to-near so the arm
   retreats away from already-placed objects.
2. **Ordering** — objects are sorted by descending distance to the target
   area centre, so the farthest object is picked first.
3. **Pose generation** — both grasp and place poses use a fixed top-down TCP
   orientation (z pointing down). The TCP z-target is set to the object's top
   surface minus a small insertion depth (GRASP_DEPTH = 0.027 m) to ensure
   stable gripper contact.

The planner has no knowledge of whether the upstream scene came from
`StatePerception` or `VisionPerception`. It operates purely on
`SceneRepresentation`.

---

### 1.3 Execution

| | |
|---|---|
| **Input** | `ActionSequence` (loaded via `executor.load(seq)`) |
| **Output** | `torch.Tensor` shape `(1, 7)` per step, passed directly to `env.step()` |

Action vector layout under `pd_ee_delta_pose` control mode:

```  
index:  0    1    2    3     4     5     6  
        dx   dy   dz   drx   dry   drz   gripper  
        ─────────────────────────────    ───────  
        EE delta pose (6-DoF, metres     +1 = open  
        and radians)                     -1 = close  
```

`MotionExecutor` runs a deterministic 9-phase state machine per object:

| # | Phase | Description |
|---|---|---|
| 1 | `OPEN_GRIPPER_INIT` | Open gripper, wait N frames |
| 2 | `PRE_GRASP` | Move to grasp point + pre-grasp height offset |
| 3 | `GRASP_DESCEND` | Descend to grasp TCP target |
| 4 | `CLOSE_GRIPPER` | Close gripper, wait N frames |
| 5 | `LIFT` | Raise to grasp_z + lift_height |
| 6 | `TRANSPORT` | Move horizontally to above place slot |
| 7 | `PLACE_DESCEND` | Descend to place TCP target |
| 8 | `OPEN_GRIPPER` | Open gripper, wait N frames |
| 9 | `RETREAT` | Raise to place_z + retreat_height |

All height offsets are relative to each action's grasp/place z, not absolute
world coordinates, so the same config works regardless of object height.

The executor does not query object state from the simulator. It checks only
TCP position (via `env.unwrapped.agent.tcp.pose`) to determine phase
transitions, with a per-phase step-count timeout as a fallback.

---

## 2. Key Design Decisions and Trade-offs

### Stability over generality

The primary design goal was a reliable, predictable baseline rather than a
system that attempts to handle every edge case. Every major decision reflects
this priority.

**Fixed top-down grasp orientation.** A single gripper orientation (z-down)
is used for all objects. This works well for compact, low-aspect-ratio objects
but is inappropriate for tall or elongated ones. During development the YCB
mustard bottle (006_mustard_bottle) was tested and rejected: its height caused
large arm motions, its footprint led to collisions with neighbouring objects,
and the top-down grasp was mechanically unsuitable. The final object set — a
primitive box, a tuna can, and an orange — was chosen to remain within the
reliable operating envelope of this fixed strategy.

**Open-loop execution.** The pipeline perceives once at episode start and
executes the full plan without re-observing. This eliminates mid-episode
sensor noise accumulation and simplifies the control loop, but means the
system cannot recover from a failed grasp. Extending to closed-loop
execution would require calling `perception.observe()` after each pick
attempt and re-invoking `planner.plan()` on the updated scene. The modular
interface is already structured to support this without any changes to
individual module internals.

**Vision perception: accuracy vs. generalisability.** For the vision-based
module, object pose is estimated from the segmented point cloud. For the YCB
orange, the visible surface is a hemisphere whose raw centroid is
systematically biased upward relative to the true sphere centre. A
sphere-fitting routine is applied specifically to this object category to
recover the correct centre position. This meaningfully improves success rate
but introduces an object-specific code path that reduces out-of-distribution
generalisation. The trade-off was accepted because the alternative — using
the biased centroid — produced consistent placement failures.

---

## 3. Adding a New Action Primitive: Pushing

Adding a push primitive requires changes to **Planning** and **Execution**
only. Perception and the `SceneRepresentation` contract remain completely
unchanged.

### Planning layer

`GraspAction` encodes only pick-and-place semantics. A new dataclass is
needed alongside it:

```python  
@dataclass  
class PushAction:  
    object_id:   str  
    push_start:  np.ndarray  # shape (3,), TCP contact point in world frame  
    push_end:    np.ndarray  # shape (3,), TCP release point in world frame  
    push_height: float       # TCP z during stroke (typically object equator)  
```

`ActionSequence.actions` would be typed as
`List[Union[GraspAction, PushAction]]`.
The planner can then decide to emit a push before a grasp when an object is
in a collision-prone position — for example, pushing it away from table edges
or other objects before attempting a top-down pick.

### Execution layer

`MotionExecutor._execute_phase()` currently assumes all actions are
`GraspAction`. A push sub-machine would introduce a parallel phase sequence:

```  
APPROACH_PUSH -> CONTACT_DESCEND -> PUSH_STROKE -> RETREAT_PUSH  
```

The gripper remains open throughout. The key structural difference from a
grasp sequence is the absence of `CLOSE_GRIPPER` and `LIFT`; instead a
horizontal `PUSH_STROKE` phase moves the TCP along the push vector at a
fixed z. The executor would dispatch to the appropriate phase list based on
`isinstance(cur_action, GraspAction)` at the start of each action.

### Summary

| Module | Change required |
|---|---|
| Perception | None |
| `SceneRepresentation` / `ObjectInfo` | None |
| `planning/base.py` | Add `PushAction` dataclass; update `ActionSequence` type annotation |
| `SequentialPlanner` | Add logic to decide when to push vs. grasp directly |
| `execution/motion_executor.py` | Add push phase state machine; dispatch on action type in `step()` |