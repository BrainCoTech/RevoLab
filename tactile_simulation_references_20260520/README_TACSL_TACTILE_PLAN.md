# RevoLab TacSL Tactile Simulation Plan

Date: 2026-05-20

This README is a handoff document for implementing tactile simulation in `BrainCoTech/RevoLab`. It summarizes the prior design discussion, the surveyed tactile simulation options, what is already present in this repository, what is missing, and the recommended implementation path.

The chosen direction is:

```text
TacSL-style Isaac Lab visuo-tactile simulation
  -> one vision-based tactile sensor per fingertip
  -> one 3D net force output per fingertip
```

For the first implementation, do **not** solve full-palm tactile skin, real-hardware calibration, full FEM, marker flow, or true material/electrical modeling. The goal is to get a working simulated tactile observation that produces useful numbers and images in RevoLab.

## 1. Final Goal

We want a Revo3 dexterous hand simulation with:

- **Five fingertip visuo-tactile sensors**, one per finger:
  - thumb
  - index
  - middle
  - ring
  - little
- **Camera-like tactile output** per fingertip:
  - tactile RGB if enabled by TacSL
  - tactile depth / distance-to-image-plane
  - optionally normal/shear force field tensors if TacSL force field is enabled
- **3D net force** per fingertip:
  - shape: `[num_envs, 5, 3]`
  - preferred final frame: fingertip local tactile frame
  - acceptable MVP frame: robot base frame or world frame, as long as it is documented

The first useful observation contract can be:

```text
tactile_rgb:       [num_envs, 5, H, W, 3] or [num_envs, 5, 3, H, W]
tactile_depth:     [num_envs, 5, H, W]
tactile_force_3d:  [num_envs, 5, 3]
```

If memory is too high, start with:

```text
tactile_depth:     [num_envs, 5, H, W]
tactile_force_3d:  [num_envs, 5, 3]
```

and enable RGB only for debugging/play mode.

## 2. What We Discussed

### Vision-based tactile sensors

Vision-based tactile sensors, such as GelSight/DIGIT-style sensors, use:

- a soft elastomer/gel surface,
- an internal camera,
- illumination,
- sometimes markers.

The camera sees deformation of the gel surface when objects contact it.

### Taxel / piezoresistive tactile skin

For normal piezoresistive tactile skin, a taxel is a tactile pixel. A physical chain could be:

```text
pressure / strain
  -> resistance change
  -> voltage divider output
  -> ADC value
```

For this project phase, we are **not** implementing this full electrical chain. We only need a simulated tactile signal that is useful for policy learning and debugging.

### Net force

TacMap outputs a net force, not a dense force map. We discussed that net force is likely a 3D vector:

```text
F = [Fx, Fy, Fz]
```

For RevoLab, we want one 3D net force vector per fingertip:

```text
five fingertips -> [5, 3]
```

This is different from a full tactile skin taxel array.

### TacMap insight

TacMap is close to the "simplified geometry-consistent representation" route:

```text
simulation:
  contact geometry / penetration depth -> deform map

real:
  tactile camera image -> learned deform map

policy:
  consumes shared deform map
```

TacMap does not output marker flow and does not output dense per-taxel force. It outputs:

```text
F: net force
P: contact position
M: deform map / penetration-depth tactile map
```

We are **not** implementing TacMap now, but its philosophy is useful: do not overbuild the physics if a consistent tactile observation is enough.

## 3. Surveyed Options And Tradeoffs

### Taccel

Taccel is a high-performance GPU tactile robotics simulator based on NVIDIA Warp and custom `warp_ipc`. It uses ABD, FEM, Neo-Hookean hyperelasticity, and IPC contact.

Pros:

- high-performance tactile robotics backend,
- strong for many contacts and soft tactile simulation,
- more physically ambitious than image-only rendering.

Cons for RevoLab now:

- not an Isaac Sim / Isaac Lab frontend,
- no ready RevoLab adapter,
- would require a custom synchronization layer from Isaac Lab states to Taccel.

Verdict:

Good long-term reference, not the first implementation path.

### Taxim

Taxim generates GelSight-style RGB from deformation/height maps using calibrated optical lookup tables.

Pros:

- good tactile RGB rendering concept,
- useful if we already have reliable height/deformation maps.

Cons:

- does not solve robot contact dynamics,
- no full Isaac Lab integration by itself,
- marker/optical output only, not a complete fingertip sensor stack.

Verdict:

Useful conceptually; TacSL already packages an Isaac Lab route that references Taxim-like rendering.

### FOTS

FOTS is a fast optical tactile simulator with marker motion modeling.

Pros:

- fast,
- marker motion support,
- good for sim-to-real tactile-motor learning.

Cons:

- marker flow is not needed in the first RevoLab stage,
- not the easiest direct integration path into current RevoLab.

Verdict:

Skip for MVP. Revisit if marker motion becomes required.

### TacEx

TacEx integrates GelSight tactile simulation into Isaac Sim / Isaac Lab using modules such as Taxim, FOTS, and GIPC/UIPC-style soft-body contact.

Pros:

- closest to "GelSight inside Isaac Sim",
- modular,
- can combine RGB, marker motion, and soft-body contact.

Cons:

- heavier than needed for "per-finger tactile + 3D net force",
- GIPC/UIPC soft-body pieces increase complexity and memory cost,
- not necessary if the goal is TacSL-style learning-friendly tactile observations.

Verdict:

Good reference if we later want more physical gel deformation or marker flow. Not the immediate path.

### TacSL

TacSL is the chosen route. NVIDIA/Isaac Lab exposes a visuo-tactile sensor built around TacSL concepts through `isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensorCfg`.

From Isaac Lab documentation, the sensor can provide:

- tactile RGB images,
- tactile depth images,
- normal force fields,
- shear force fields.

Key config concepts from the official docs:

```python
from isaaclab.sensors import TiledCameraCfg
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

tactile_sensor = VisuoTactileSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
    render_cfg=GELSIGHT_R15_CFG,
    enable_camera_tactile=True,
    enable_force_field=True,
    tactile_array_size=(20, 25),
    tactile_margin=0.003,
    contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
    normal_contact_stiffness=1.0,
    friction_coefficient=2.0,
    tangential_stiffness=0.1,
    camera_cfg=TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
        update_period=1 / 60,
        height=320,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=None,
    ),
)
```

Important requirements from the official Isaac Lab docs:

- `enable_camera_tactile=True` needs a valid `TiledCameraCfg`.
- `enable_force_field=True` needs contact objects with SDF collision meshes.
- The sensor `prim_path` must be a child of the elastomer prim in the USD hierarchy.
- Query points for the force field are computed from the elastomer mesh.
- Camera settings currently use `distance_to_image_plane`.
- Object prims must exist before simulation initialization so the SDF view can be created.

Verdict:

This is the best route for RevoLab now because the repo already uses Isaac Lab, and TacSL is designed for visuo-tactile observations in learning workflows.

### DiffTactile

DiffTactile is a differentiable tactile simulator with physics-based elastomer simulation.

Pros:

- differentiable,
- physically meaningful,
- useful for parameter identification and contact-rich optimization.

Cons:

- not the immediate Isaac Lab integration path,
- heavier than needed for a first working RevoLab tactile observation.

Verdict:

Useful research reference; not MVP.

### IPC / GIPC / UIPC / TacIPC

These are high-fidelity contact/soft-body simulation routes.

Pros:

- robust deformable contact,
- good for soft tactile physics.

Cons:

- integration cost is high,
- not needed for first TacSL-style fingertip sensor.

Verdict:

Do not start here.

## 4. What RevoLab Already Has

Repository cloned at:

```text
/Users/xulixin/revolab_tactile_work/RevoLab
```

Current inspected commit:

```text
bc2f874
```

RevoLab is already an Isaac Lab extension package. It includes:

- Revo3 USD and URDF assets,
- Direct RL environments,
- ManagerBased RL environments,
- HORA rotation environments,
- Dexsuite grasp/lift environments,
- RSL-RL, RL-Games, and HORA training scripts,
- deployment code for Revo3.

### Existing contact force support

The repo already enables contact sensors on the robot assets:

```python
activate_contact_sensors=True
```

Examples:

- `source/BrainCo_DexHand/BrainCo_DexHand/assets/tianji_revo3_right.py`
- `source/BrainCo_DexHand/BrainCo_DexHand/assets/revo3_repose.py`
- `source/BrainCo_DexHand/BrainCo_DexHand/assets/revo3_reorient.py`

Dexsuite grasp config already attaches one `ContactSensorCfg` per DIP link:

```python
for link_name in TIANJI_HAND_DIP_BODIES:
    setattr(
        self.scene,
        f"{link_name}_object_s",
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        ),
    )
```

File:

```text
source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/Revo3/dexsuite_revo3_env_cfg_grasp.py
```

It also exposes a contact observation:

```python
self.observations.proprio.contact = ObsTerm(
    func=mdp.fingers_contact_force_b,
    params={"contact_sensor_names": [f"{link}_object_s" for link in TIANJI_HAND_DIP_BODIES]},
    clip=(-20.0, 20.0),
)
```

The observation helper already returns 3D forces concatenated across fingertips:

```python
def fingers_contact_force_b(...):
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b.view(env.num_envs, -1)
```

File:

```text
source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/observations.py
```

HORA also already computes smoothed contact force magnitudes from `net_forces_w_history`:

```python
net_contact_forces_history = torch.cat([
    self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2)
    for id in self._contact_body_ids
], dim=2)
norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
```

File:

```text
source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_rotation/revo3_hand_hora_env.py
```

This means the `3D net force` part is already close. What is missing is a clean, reusable tactile sensor module and TacSL visuo-tactile integration.

## 5. What Is Missing

### Missing 1: TacSL / Isaac Lab contrib dependency check

The repo currently does not contain:

```text
isaaclab_contrib.sensors.tacsl_sensor
VisuoTactileSensorCfg
GELSIGHT_R15_CFG / GELSIGHT_MINI_CFG usage
```

The implementing agent must first verify the installed Isaac Lab version includes:

```python
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
from isaaclab_assets.sensors import GELSIGHT_R15_CFG, GELSIGHT_MINI_CFG
```

If unavailable, the Isaac Lab version or extension set must be updated.

### Missing 2: fingertip elastomer USD structure

Official TacSL sensor config expects something like:

```text
Robot/
  right_index_elastomer/
    tactile_sensor
    cam
```

or at least a sensor prim that is a child of an elastomer prim.

Current RevoLab USD assets appear to have standard robot links, not dedicated TacSL elastomer/camera prims. Because `pxr` is not available in the current shell, this needs verification inside the Isaac Lab Python environment.

Need to inspect or create USD hierarchy for each fingertip:

```text
right_thumb_tactile_elastomer
right_index_tactile_elastomer
right_middle_tactile_elastomer
right_ring_tactile_elastomer
right_little_tactile_elastomer
```

Each should have:

- an elastomer mesh surface,
- collision/contact properties,
- a child tactile sensor prim,
- a child camera prim if camera tactile is enabled.

### Missing 3: SDF collision meshes for contacted objects

TacSL force field computation requires SDF collision meshes for interacting objects. The first implementation should only support a small set of objects:

- cube,
- cylinder,
- ball.

Need to ensure their USD/PhysX collision setup supports SDF queries. If not, create/convert SDF-compatible collision assets for these objects.

### Missing 4: per-fingertip TacSL sensor configs

Need one `VisuoTactileSensorCfg` per fingertip.

Suggested names:

```text
thumb_tactile_sensor
index_tactile_sensor
middle_tactile_sensor
ring_tactile_sensor
little_tactile_sensor
```

Suggested first config:

```text
enable_camera_tactile=True
enable_force_field=False initially if SDF setup blocks progress
```

Then enable force field after SDF objects are verified:

```text
enable_force_field=True
```

Start with low resolution:

```text
camera height/width: 64x64 or 80x60 for training
tactile_array_size: (16, 16) or (20, 25)
num_envs: 1-16 for debugging
```

Only scale up after memory and runtime are understood.

### Missing 5: unified 3D net-force readout

Current code has two styles:

```text
Dexsuite: force_matrix_w -> base-frame 3D force
HORA: net_forces_w_history -> magnitude only
```

Need a unified output:

```text
tactile_force_3d = [num_envs, 5, 3]
```

MVP source:

```text
Isaac Lab ContactSensor
```

Later source:

```text
integrate TacSL normal/shear force fields over each fingertip surface
```

For the first version, keep ContactSensor as the authoritative 3D net force. This is simple, robust, and already mostly present.

### Missing 6: local force frame

The best representation is fingertip-local:

```text
Fx, Fy, Fz in each tactile sensor frame
```

Current Dexsuite helper transforms force to robot base frame. That is acceptable for MVP, but the better implementation should transform to each fingertip/tactile frame using the fingertip body quaternion.

Need output options:

```text
force_w      # world frame
force_b      # robot base frame
force_local  # fingertip tactile frame
```

### Missing 7: observation integration

Need to decide which environment gets TacSL first.

Recommended first target:

```text
HORA rotation environment
```

Reason:

- already has 5 contact sensors,
- already uses contact history as tactile observation,
- one hand and one object,
- easier to debug than full manager-based task.

Second target:

```text
Dexsuite Revo3 lift/grasp
```

Reason:

- already has `ContactSensorCfg` in manager-based config,
- manager-based observation terms make it natural to add tactile observation terms.

### Missing 8: debug/demo script

Need a no-training smoke test:

```text
python scripts/debug_tactile/tacsl_revo3_smoke.py --num_envs 1 --finger index
```

It should:

- spawn hand and object,
- enable one tactile sensor first,
- move/contact object or start from a known contact pose,
- print tensor shapes,
- print 3D force values,
- optionally save tactile RGB/depth images.

Do not start by training.

## 6. Recommended Implementation Stages

### Stage 0: preserve existing ContactSensor force

Goal:

```text
Expose clean [num_envs, 5, 3] net force.
```

Tasks:

1. Add a small utility/helper for net force readout.
2. Reuse existing `ContactSensorCfg` on five DIP links.
3. Return force in world/base/local frame.
4. Add smoke test that prints force when object contacts fingertips.

Expected output:

```text
tactile_force_3d: [num_envs, 5, 3]
```

No TacSL yet in this stage.

### Stage 1: one fingertip TacSL sensor

Goal:

```text
Get one fingertip visuo-tactile sensor running.
```

Use only one finger first, preferably index:

```text
right_index_DIP_Link
```

Tasks:

1. Verify Isaac Lab has `VisuoTactileSensorCfg`.
2. Inspect/create USD elastomer and camera prim for index fingertip.
3. Add one sensor to the scene config.
4. Enable camera tactile depth first.
5. Save/print output tensor shapes.

Expected output:

```text
index_tactile_depth: [num_envs, H, W]
index_tactile_rgb: optional
index_force_3d: [num_envs, 3]
```

### Stage 2: five fingertip TacSL sensors

Goal:

```text
All five fingers have TacSL tactile output.
```

Tasks:

1. Generalize index sensor config to five fingers.
2. Keep resolution low.
3. Stack outputs along finger dimension.
4. Add config flags:

```python
enable_tacsl_tactile = True
enable_tactile_rgb = False
enable_tactile_depth = True
enable_tactile_force_field = False
tactile_image_height = 64
tactile_image_width = 64
```

Expected output:

```text
tactile_depth: [num_envs, 5, H, W]
tactile_force_3d: [num_envs, 5, 3]
```

### Stage 3: TacSL force field

Goal:

```text
Enable normal/shear force field from TacSL.
```

Tasks:

1. Ensure contact objects have SDF collision meshes.
2. Enable `enable_force_field=True`.
3. Read:

```python
tactile_data.tactile_normal_force
tactile_data.tactile_shear_force
```

4. Compare integrated force field against ContactSensor force.

Expected outputs:

```text
tactile_normal_force: [num_envs, 5, ...]
tactile_shear_force:  [num_envs, 5, ...]
tactile_force_3d_contact_sensor: [num_envs, 5, 3]
tactile_force_3d_from_field: optional
```

Keep ContactSensor force as fallback.

### Stage 4: policy observation integration

Goal:

```text
Expose tactile observation to RL policies.
```

For low-dimensional policy:

```text
use tactile_force_3d only
```

For visual tactile policy:

```text
use tactile_depth or tactile_rgb encoder
```

Initial policy observation can be:

```text
joint positions
joint targets
object state
tactile_force_3d flattened
```

Do not immediately train with high-resolution RGB across thousands of envs.

## 7. Suggested File Layout

Add new files:

```text
source/BrainCo_DexHand/BrainCo_DexHand/sensors/__init__.py
source/BrainCo_DexHand/BrainCo_DexHand/sensors/tactile_net_force.py
source/BrainCo_DexHand/BrainCo_DexHand/sensors/tacsl_fingertip.py
scripts/debug_tactile/tacsl_revo3_smoke.py
```

Possible config additions:

```text
source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_rotation/revo3_hand_hora_env_cfg.py
source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_rotation/revo3_hand_hora_env.py
source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/config/Revo3/dexsuite_revo3_env_cfg_grasp.py
source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/observations.py
```

Possible asset additions:

```text
assets/usd/tactile/
assets/usd/tactile/revo3_right_tacsl_fingertips.usd
```

or integrate directly into existing Revo3 USD assets, if that is cleaner.

## 8. Important Design Decisions

### Use ContactSensor for 3D net force first

Even though TacSL can output force fields, the easiest reliable 3D net force is existing Isaac Lab `ContactSensor`.

Why:

- already used in RevoLab,
- already provides 3D vectors,
- cheap,
- works before SDF force field setup is solved.

Later, compare it with integrated TacSL force fields.

### Use TacSL for visuo-tactile output

Do not hand-roll a fake camera or Taxim clone in RevoLab. Use Isaac Lab's TacSL-style `VisuoTactileSensorCfg` if available.

### Start with fingertips only

Do not implement palm or full-hand skin now.

Initial tactile bodies:

```text
right_thumb_DIP_Link
right_index_DIP_Link
right_middle_DIP_Link
right_ring_DIP_Link
right_little_DIP_Link
```

Full palm/finger pad tactile skin can come later.

### Start with depth before RGB

RGB tactile output is expensive and may require more asset/rendering tuning. Depth/distance-to-image-plane is enough to verify the camera tactile path.

### Keep force output low-dimensional

The immediate required force output is:

```text
one 3D net force per finger
```

Do not start with dense taxel maps.

## 9. Risks And Known Unknowns

### Isaac Lab version compatibility

The official visuo-tactile sensor docs are from Isaac Lab docs updated around 2026-05-19. RevoLab's required Isaac Lab version must match APIs such as:

```python
isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensorCfg
isaaclab_assets.sensors.GELSIGHT_R15_CFG
```

If the local Isaac Lab install is older, imports may fail.

### USD asset preparation

TacSL expects elastomer/camera prims in the USD hierarchy. Current RevoLab assets may not have those. This is likely the biggest work item.

### SDF force field setup

TacSL force fields require SDF collision meshes on contact objects. If objects are not SDF-ready, force field output may fail even if camera tactile works.

### Memory and performance

Five tactile cameras across many environments can be expensive.

Start with:

```text
num_envs = 1-16
resolution = 64x64
RGB disabled
force field disabled
```

Then scale.

### Observation shape changes break checkpoints

Existing checkpoints expect specific observation dimensions. Adding tactile observations will break old policies unless the observation is optional or a new task/config name is registered.

Use separate task variants:

```text
BrainCo-Direct-Revo3-HoraRotate-Cylinder-Tactile-v0
BrainCo-Dexsuite-Revo3-Right-Lift-Tactile-v0
```

rather than changing existing public task IDs.

## 10. Acceptance Criteria

### MVP acceptance

1. A smoke test runs with one environment.
2. It prints:

```text
tactile_force_3d.shape == [1, 5, 3]
```

3. At least one fingertip force becomes nonzero on contact.
4. One fingertip tactile depth image is produced and can be saved.

### Stage 2 acceptance

1. Five fingertip tactile depth streams exist.
2. Five 3D force vectors exist.
3. Output tensors have stable shapes:

```text
tactile_depth.shape == [num_envs, 5, H, W]
tactile_force_3d.shape == [num_envs, 5, 3]
```

4. No NaNs.
5. Works with `num_envs=16`.

### Stage 3 acceptance

1. TacSL force field output is enabled for SDF-compatible objects.
2. Integrated force field roughly correlates with ContactSensor net force.
3. ContactSensor remains available as fallback.

## 11. References

- Isaac Lab Visuo-Tactile Sensor docs: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/visuo_tactile_sensor.html
- TacSL paper/project: https://arxiv.org/abs/2408.06506
- TacMap paper: https://arxiv.org/pdf/2602.21625
- TacEx paper/code: https://arxiv.org/abs/2411.04776, https://github.com/DH-Ng/TacEx
- Taccel paper/code: https://arxiv.org/abs/2504.12908, https://github.com/Taccel-Simulator/Taccel
- Taxim paper/code: https://arxiv.org/abs/2109.04027, https://github.com/CMURoboTouch/Taxim
- FOTS paper/code: https://arxiv.org/abs/2404.19217, https://github.com/Rancho-zhao/FOTS
- DiffTactile paper/project: https://arxiv.org/abs/2403.08716, https://difftactile.github.io/

## 12. One-Sentence Implementation Brief

Implement a Revo3 fingertip tactile sensor stack in Isaac Lab using the TacSL-style `VisuoTactileSensorCfg` for per-finger tactile depth/RGB, and reuse or wrap Isaac Lab `ContactSensor` to expose one 3D net force vector per fingertip, starting with the HORA or Dexsuite Revo3 environments and keeping the feature behind new tactile-specific config/task variants.
