# Modular Pick and Place

A modular robotic manipulation system for pick-and-place tasks, built on
[ManiSkill](https://github.com/haosulab/ManiSkill) with a Franka Panda arm.
The system is decomposed into three independently swappable modules —
**Perception**, **Planning**, and **Execution** — connected through
well-defined data interfaces.

---

## Repository Structure

```  
.  
├── config/  
│   └── scene.yaml              # Object set and target area definition  
├── envs/  
│   ├── multi_object_env.py     # ManiSkill env: MultiObjectPickAndPlace-v1  
│   └── scene_config.py         # YAML loader and scene parameter dataclasses  
├── perception/  
│   ├── base.py                 # SceneRepresentation, ObjectInfo, TargetArea, BasePerception  
│   ├── state_perception.py     # Privileged-state implementation  
│   └── vision_perception.py    # RGB-D + segmentation implementation  
├── planning/  
│   ├── base.py                 # GraspAction, ActionSequence, BasePlanner  
│   └── sequential_planner.py   # Distance-sorted greedy planner  
├── execution/  
│   ├── base.py                 # ExecutorConfig interface  
│   └── motion_executor.py      # 9-phase deterministic state machine  
├── tests/                      # Per-module unit tests  
├── demo.py                     # Full pipeline runner  
├── DESIGN.md                   # Module interfaces and design rationale  
└── README.md  
```

---

## Prerequisites

- Python 3.8+
- [ManiSkill](https://github.com/haosulab/ManiSkill) (`pip install mani-skill`)
- GPU recommended for rendering
- `imageio`, `Pillow` for video output

```bash  
pip install mani-skill imageio Pillow  
```

---

## Quick Start

**State-based perception, 10 episodes:**

```bash  
python demo.py --perception state --episodes 10  
```

**Vision-based perception, 10 episodes:**

```bash  
python demo.py --perception vision --episodes 10  
```

**Custom config or episode count:**

```bash  
python demo.py --config config/scene.yaml --perception vision --episodes 5  
```

Each run prints per-object success/failure for every episode and a final
success rate summary. A video of all episodes is saved automatically:

- `demo_state_all_episodes.mp4`
- `demo_vision_all_episodes.mp4`

---

## Scene Configuration

Defined in `config/scene.yaml`. The default scene contains three objects from
different categories:

| Object ID | Category | Description |
|---|---|---|
| `box_0` | Primitive box | 5 cm cube, red |
| `obj_tuna` | YCB mesh | `007_tuna_fish_can` |
| `obj_orange` | YCB mesh | `017_orange` |

Objects are spawned at randomised positions on every reset:

- x ∈ [−0.30, −0.15] m
- y ∈ [−0.15, 0.15] m
- minimum inter-object distance: 0.08 m

The target area is an 18 cm × 16 cm region centred at (0.02, 0.0) m,
highlighted in semi-transparent yellow on the table surface.

---

## Architecture Overview

```  
┌──────────────┐  SceneRepresentation  ┌──────────────┐  ActionSequence  ┌──────────────┐  
│  Perception  │ ─────────────────────▶│   Planning   │ ───────────────▶ │  Execution   │  
│              │                       │              │                  │              │  
│  obs: dict   │                       │  (no env     │                  │  env.step()  │  
│  state/RGB-D │                       │   access)    │                  │  tensor(1,7) │  
└──────────────┘                       └──────────────┘                  └──────────────┘  
```

The only connection between modules is data. No module holds a reference to
another module's internals or queries the simulator outside its defined input.

Swapping `StatePerception` for `VisionPerception` requires changing one line
in `demo.py`. Planning and Execution are completely unaffected — this is the
direct result of the shared `SceneRepresentation` contract.

---

## Evaluation Results

All results are from 10 randomised episodes per run. Each episode resets
object positions with a different seed.

### State-Based Perception

| Metric | Value |
|---|---|
| Best run success rate | **10 / 10 (100%)** |
| Typical success rate | 80 – 90% |

### Vision-Based Perception

| Metric | Value |
|---|---|
| Best run success rate | **9 / 10 (90%)** |
| Typical success rate | 70 – 80% |

The vision-based pipeline performs slightly below the state-based baseline,
primarily due to point-cloud pose estimation noise. The gap is most visible
for the YCB orange: a sphere-fitting step is applied to correct the
systematic centroid bias caused by the object's partial occlusion against
the table surface.

---

## Running Tests

```bash  
python -m pytest tests/ -v  
```

---

## Design

See [DESIGN.md](DESIGN.md) for full documentation of:

- Module interfaces (inputs/outputs and data schemas for each module)
- Key design decisions and trade-offs
- What would need to change to add a new action primitive (e.g. pushing)