# Task scene audit after the asset migration

The 2026-10-17 audit compared the original README demonstrations and task assets
at `bc2f874` with the pinned description assets at `6db7e2e`. A matching USD hash
and joint space did not establish matching task geometry. The earlier migration
smokes checked loading and finite stepping, but missed the arm placement errors.

| Task | Finding and disposition |
| --- | --- |
| Handover | Shoulder mount and wrist frames changed. Old angles raised both hands above the head; the new support base also extended below the ground. Refit both arms against the original shoulder-relative elbow/palm poses and place the waist-frame root at z=1.15 m. Retrain from scratch. |
| Lift | Single-arm mount turned the workspace 90 degrees away from the table. Set Joint1_R to -pi/2. Also exclude the overlapping Link5_R/Link7_R wrist housings from mutual collision. Retrain from scratch. |
| Repose | Palm still faces upward, with the cube above the palm/fingers. A held reset settles the cube into the hand. No corresponding scene mismatch found; resume the interrupted v1 run. |
| Reorient | Same upward palm orientation and cylinder placement as the original task. Held reset retains the cylinder. Keep the existing v1 training run. |
| HORA Ball | Palm retains its 25-degree tilt and the ball lies between the fingertips. New-asset grasp cache already validated separately at normal gravity. Keep the existing local Stage 1 run. |
| HORA Cylinder | Same tilted palm and cylindrical grasp layout. Generate and validate a new-asset cache before starting Stage 1. |

## Arm pose evidence

Handover was fitted using joint-frame forward kinematics from the original USD
and current USD, with the shoulder origins aligned. The fitted right/left palm
position errors are 0.35/0.35 mm, orientation errors 0.069/0.068 degrees, and elbow
position errors 1.77/1.79 mm. These are geometric comparisons, not tracking errors
or policy success measurements. The new shell/body appearance is retained.

Isaac Sim reset positions agree with the new kinematics. Palm outward normals
(the palm frame's +X axis) are approximately `(0.100, 0.422, 0.901)` on the right
and `(0.105, -0.420, 0.901)` on the left. Before repair their world Z component was
`-0.105`. The current asset's support base reaches z=-1.14948 m in asset
coordinates; root z=1.15 m puts its bottom just above the ground.

Lift's corrected palm is approximately `(0.575, 0.0, 1.296)` m, facing downward.
The original palm frame was at `(0.586, 0.0, 1.294)` m. The new palm frame differs
by about 12 mm, while the arm shape and downward orientation remain aligned.
Before correction the palm was at `(1.0, 0.425, 1.296)` m.

A second Lift failure appeared during a 60-physics-step hold: self contact between
Link5_R and Link7_R displaced the palm by about 0.27 m. The contact-force matrix
identified this wrist pair. Joint6_R and Joint7_R share an origin and the housing
colliders overlap. Filtering only this pair removes the impulse; the same hold
then retains the initial palm pose. All other self collisions and all object
contacts remain enabled. The exclusion is authored on each spawned task instance,
not in the upstream asset submodule.

## Reproduction and limits

Run `scripts/smoke_tasks.py` with the desired v1 task on a prepared simulator.
It now asserts reset palm directions/positions and checks the Lift wrist exclusion
and stability under zero relative actions, in addition to finite stepping and
checkpoint metadata validation. Use at least two environments to check clones.

```bash
bash scripts/remote/run.sh 1 scripts/smoke_tasks.py \
  --task BrainCo-Dynamic-Handover-Revo3-Cube-v1 --num_envs 2 --steps 32
bash scripts/remote/run.sh 1 scripts/smoke_tasks.py \
  --task BrainCo-Dexsuite-Revo3-Right-Lift-v1 --num_envs 2 --steps 32
```

Arm-task checkpoint contracts now include the robot's configured initial state.
The prior Handover/Lift runs must not be resumed into the repaired scenes.
Unaffected in-hand tasks retain their existing contract and resume compatibility.

Actual renders, original GIF extracts, kinematic comparisons, contact probes and
hold traces from this audit are under ignored `tmp/scene-repair/20261017/`.
These checks validate scene initialization, not trained manipulation success.
HORA's existing weak-gravity curriculum behavior remains a separate training
quality issue; no curriculum or reward changes are included in this repair.
