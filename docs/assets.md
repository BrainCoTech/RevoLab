# Robot asset contract

The only robot-geometry source is `third_party/brainco-description`, pinned to
`6db7e2ed1d864ee0a3ab6ff62d43a303787137f6` (20261014 release). RevoLab does not patch or copy its USD/URDF/mesh files.

| Consumer | Upstream USD | Moving joints | Rigid bodies | USD collider prims |
| --- | --- | ---: | ---: | ---: |
| Repose, Reorient, both HORA tasks | `revo3_system/usd/revo3_right.usd` | 21 | 22 | 82 |
| Lift | `revotron_system/usd/revotron_single_right_revo3.usd` | 28 | 29 | 92 |
| Handover | `revotron_system/usd/revotron_bimanual_revo3.usd` | 59 | 60 | 189 |

These are authored USD collider counts, not the number of collision shapes after
PhysX cooking. Convex decomposition can produce many shapes per collider: the
current Reorient hand creates 833 runtime shapes from its 82 USD colliders.
Check runtime shape counts and sampling throughput when updating collision assets.
Of those 833 shapes, 81 belong to the palm/wrist body. The
[palm collision experiment](palm-collision-experiment.md) inspects their source
and provides a 41-hull trial with measured surface/contact differences; it has
not replaced the production asset.

The corresponding old USD collider counts were 49, 57 and 114. Changes include finger axes and limits, inertial frames, collision geometry and fixed-joint merging. These changed the transition dynamics and observations of every published task and required all six policies to be retrained. The subsequent 2026-10-19 campaign produced successful simulation demos for all six tasks, including HORA Stage 1 with new grasp caches. HORA Stage 2 remains untrained. See the [current results](../README.md#simulation-demos).

## Task adapters

The [2026-10-17 scene audit](task-scene-validation.md) corrects Handover and Lift
initial poses that the initial loading checks missed. Those two tasks were
retrained after correction; checkpoints from their earlier incorrect v1 scenes
must not be resumed in the corrected scenes.

- `assets/description.py` resolves the submodule paths and detects missing files/LFS pointers.
- `assets/frames.py` computes named fixed-frame poses from their nearest rigid ancestor. Linear velocity includes the angular-velocity cross offset term. Missing names raise errors; there is no root-body fallback.
- Finger rigid links use the upstream `_link` spelling. Palm and tip Xforms retain names such as `right_palm` and `right_index_tip_Link`.
- Handover locks `waist_pitch_Joint`, `neck_yaw_Joint`, and `head_pitch_Joint` as native fixed joints at their zero configuration in every spawned scene. The source USD has 59 moving joints; the task articulation has 56 DOFs and 56 actions.
- Lift excludes mutual contact between the overlapping `Link5_R`/`Link7_R` wrist housings in the spawned scene. Other self collisions and object collisions remain enabled.
- Task actuator gains and hand effort/velocity caps are explicit simulation choices. They are not calibrated hardware specifications. Arm torque caps follow the new URDF's 108/66/18 Nm groups.
- `_mx_touch` variants and their extra sample frames are not used by these tasks. Contact observation semantics are unchanged unless documented in a task update.

## Artifacts

No v0 model or cache is accepted as a v1 artifact. Each run stores `params/asset_contract.json`; it identifies the task, asset commit/path, actual simulator joint order, physical and control limits, USD SHA-256, action joints, and observation/action spaces. Training configuration YAML records gains, rewards and randomization settings. Evaluation and resumption compare the contract before loading weights.
Handover and Lift also record the configured robot initial position, rotation and
joint angles, so matching USD hashes cannot accept checkpoints from their prior
incorrect scenes.

HORA `--cache_file` accepts a `.npz` file containing:

- `asset_revision`: scalar string equal to the pinned commit.
- `joint_names`: ordered string array matching the simulator articulation's 21 joints.
- `states`: nonempty float array `[N, 28]`; 21 joint positions followed by object XYZ and WXYZ quaternion in the environment frame.

Rows must be finite, within the task's scaled limits, and use normalized quaternions. Omitting the cache uses the configured default pose for startup checks. This pose has not been validated as a successful grasp distribution.

Generate a cache against the pinned asset in the Isaac Lab environment:

```bash
python scripts/hora/generate_grasp_cache.py --task ball --output /path/to/ball-v1.npz --headless
```

The sampler perturbs the configured grasp and retains initial states that hold
the object for six seconds at Earth gravity without resetting, with at least
two object-filtered fingertip contacts during the final hold window. It replays
accepted candidates with newly randomized PD gains before saving unique rows.
Task mass, friction and center-of-mass randomization remain enabled; external
force disturbances are disabled for this initialization check. The adjacent JSON
records trial counts and thresholds. A cache passing this check supplies training
resets; it does not establish rotation skill or robustness to training disturbances.

## Updating the submodule

1. Review upstream joint, inertia, collision and frame changes.
2. Check out the desired upstream commit, fetch LFS, and update `DESCRIPTION_REVISION`.
3. Update adapters and explicitly version any changed policy contract.
4. Run `scripts/check_assets.py` and all six `scripts/smoke_tasks.py` tasks. Validate frame placement and joint locking in the simulator.
5. Record retraining/calibration requirements before publishing weights or demonstrations.

The upstream release has no LICENSE/COPYING file. RevoLab's MIT license does not grant rights to those assets; the owning team still needs to publish the asset license upstream.
