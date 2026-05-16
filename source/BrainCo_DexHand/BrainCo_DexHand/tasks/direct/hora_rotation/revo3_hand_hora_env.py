# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import os

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import isaaclab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_conjugate, quat_mul, axis_angle_from_quat, saturate

if TYPE_CHECKING:
    from .revo3_hand_hora_env_cfg import Revo3HandHoraEnvCfg


class Revo3HandHoraEnv(DirectRLEnv):
    """DirectRLEnv for Revo3 right hand in-hand object rotation.

    Observation (141 dims) — 3-frame sliding window, 47 dims/frame:
      [0:21]   joint positions, unscaled to [-1,1] via (2x - hi - lo)/(hi - lo), +-0.02 rad noise
      [21:42]  current joint targets (delta-accumulated, clamped to joint limits)
      [42:47]  contact forces on 5 DIP fingertips (smoothed alpha=0.5, latency p=0.005)

    Action (21 dims) — delta position control:
      action ∈ [-1,1] → target = prev_target + (1/24)*action → clamp(joint_limits)
      Torque control: torque = p_gain*(target - pos) - d_gain*vel
      p_gain/d_gain from cfg (2.0/0.2), randomized per reset: ×[0.5, 2.0] per-DOF

    Reward (6 terms, total ×0.01 for PPO):
      rotate:    clip(angvel·Z_axis, -0.5, 0.5) × 2.5      — encourage Z-axis rotation
      linvel:    ‖obj_Δpos‖₁ / dt × (-0.3)                  — suppress translation
      obj_pos:   1/(‖obj_pos - init_pos‖ + 0.001) × 0.003   — stay near initial position
      pos_diff:  Σ(joint_pos - init_joint_pos)² × (-0.4)    — stay near grasp pose from assets.py
      torque:    Σ(self.torques²) × (-0.1)                   — suppress excessive PD torques
      work:      (Σ(self.torques × vel))² × (-0.5)           — suppress mechanical power

    Termination:
      height:    obj_z outside [init_z - 2cm, init_z + 2cm]
      timeout:   episode_length >= max_episode_length (400 steps @20Hz)
      gravity curriculum: when height reset rate < 0.05%, gravity += 0.05 from -0.05 up to 10

    Key design decisions:
      - init_joint_pos from assets.py is the single source of truth for grasp pose reference
      - PD gains hardcoded from cfg.pgain/dgain, not read from URDF/USD
      - torque/work penalty uses self.torques (our explicit PD command), not PhysX applied_torque
      - Stage2: enable_contact_in_obs=False zeros contacts in actor obs; proprio_hist retains real contacts
    """
    cfg: Revo3HandHoraEnvCfg

    def __init__(self, cfg: Revo3HandHoraEnvCfg, render_mode: str | None = None, **kwargs):
        self.reset_height_lower = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.reset_height_upper = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # Canonical init joint pose from assets.py — used for pos_diff_penalty and cache-less reset
        self.init_joint_pos = torch.zeros((1, self.num_hand_dofs), device=self.device)
        _cfg_pos = self.cfg.robot_cfg.init_state.joint_pos
        if _cfg_pos:
            for _name, _val in _cfg_pos.items():
                if _name in self.hand.joint_names:
                    self.init_joint_pos[0, self.hand.joint_names.index(_name)] = float(_val)

        self._axes_visualizer = None
        if getattr(self.cfg, 'debug_show_axes', True):
            try:
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import FRAME_MARKER_CFG
                # create frame marker configuration for cylinder
                axes_marker_cfg = FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/CylinderAxes"
                )
                # adjust the axes size based on config (default 0.06 m)
                axes_length = getattr(self.cfg, 'vis_cylinder_axes_length', 0.06)
                axes_marker_cfg.markers["frame"].scale = (axes_length, axes_length, axes_length)
                # create the visualization marker
                self._axes_visualizer = VisualizationMarkers(axes_marker_cfg)
            except Exception as e:
                self._axes_visualizer = None

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # buffers for object
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_pos_prev = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot_prev = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_default_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.rb_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # buffers for data
        self.obs_buf_lag_history = torch.zeros((self.num_envs, 80, self.cfg.observation_space//3), device=self.device, dtype=torch.float)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.proprio_hist_buf = torch.zeros((self.num_envs, self.cfg.prop_hist_len, self.cfg.observation_space//3), device=self.device, dtype=torch.float)
        self.priv_info_buf = torch.zeros((self.num_envs, self.cfg.priv_info_dim), device=self.device, dtype=torch.float)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0] * self.cfg.dof_limits_scale
        self.hand_dof_upper_limits = joint_pos_limits[..., 1] * self.cfg.dof_limits_scale

        # Hardcoded PD gains — not reading from URDF/USD baked-in defaults
        ndof = self.num_hand_dofs
        self.p_gain = torch.ones((self.num_envs, ndof), device=self.device) * self.cfg.pgain
        self.d_gain = torch.ones((self.num_envs, ndof), device=self.device) * self.cfg.dgain

        # grasp_cache
        self.scale_ids = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int32)
        cache_path = f"{self.cfg.grasp_cache_path}.npy"
        if os.path.exists(cache_path):
            self.saved_grasping_states = torch.from_numpy(np.load(cache_path)).float().to(self.device)
            self.bucket_grasp = self.saved_grasping_states.shape[0]
            self.bucket_env = self.num_envs
        else:
            print(f"[WARN] Grasp cache not found: {cache_path}, falling back to default pose.")
            self.saved_grasping_states = None

        self.rot_axis = torch.tensor(self.cfg.rot_axis, dtype=torch.float32).repeat(self.num_envs, 1).to(self.device)

        # contact buffers
        self._contact_body_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self._contact_body_ids_disable = torch.tensor(self.cfg.disable_tactile_ids, dtype=torch.long)
        self.last_contacts = torch.zeros((self.num_envs, len(self._contact_body_ids)), dtype=torch.float, device=self.device)
        self.elastomer_ids = [self.hand.body_names.index(body_name) for body_name in self.cfg.elastomer_body_names]

        # randomize
        if self.cfg.randomize_friction:
            rand_friction = torch.empty(self.num_envs).uniform_(self.cfg.randomize_friction_scale_lower, self.cfg.randomize_friction_scale_upper)
            rand_friction = rand_friction.reshape(self.num_envs, 1)
            rand_friction_object = rand_friction.clone() * self.cfg.object_base_friction
            self.set_friction(self.object, rand_friction_object, self.num_envs)
            n_hand_mats = self.hand.root_physx_view.get_material_properties().shape[1]
            rand_friction_hand = rand_friction.clone().repeat(1, n_hand_mats) * self.cfg.metal_base_friction
            self.set_friction(self.hand, rand_friction_hand, self.num_envs)
            self.priv_info_buf[:, 3] = rand_friction.squeeze()
        if self.cfg.randomize_com:
            rand_com = torch.empty([self.num_envs, 3]).uniform_(self.cfg.randomize_com_lower, self.cfg.randomize_com_upper)
            self.set_com(self.object, rand_com, self.num_envs)
            self.priv_info_buf[:, 5:8] = self.object.root_physx_view.get_coms().reshape(self.num_envs, -1)[:, :3]
        if self.cfg.randomize_mass:
            rand_mass = torch.empty(self.num_envs).uniform_(self.cfg.randomize_mass_lower, self.cfg.randomize_mass_upper)
            self.set_mass(self.object, rand_mass, self.num_envs)
            self.priv_info_buf[:, 4] = self.object.root_physx_view.get_masses().reshape(self.num_envs)

        # physics_sim_view
        self.physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # contact sensors
        self._contact_sensor = []
        for id in range(len(self.cfg.contact_sensor)):
            self._contact_sensor.append(ContactSensor(self.cfg.contact_sensor[id]))
            self.scene.sensors[f"contact_sensor_{id}"] = self._contact_sensor[id]
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Delta position control: action ∈ [-1,1] → target += (1/24)*action → clamp to joint limits.
        Also updates object_rot_prev/pos_prev for angular velocity computation in reward."""
        actions = saturate(actions, torch.tensor(-self.cfg.clip_actions), torch.tensor(self.cfg.clip_actions))
        self.actions = actions.clone()
        targets = self.prev_targets + self.cfg.action_scale * self.actions
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            targets,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.object_pos_prev[:] = self.object_pos
        self.object_rot_prev[:] = self.object_rot

        if self.cfg.force_scale > 0.0:
            self.rb_forces *= torch.pow(torch.tensor(self.cfg.force_decay, dtype=torch.float32), self.physics_dt / self.cfg.force_decay_interval)
            # apply new forces
            obj_mass = self.object.root_physx_view.get_masses().reshape(self.num_envs).to(self.device)
            prob = self.cfg.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero().to(self.device)
            self.rb_forces[force_indices, :] = torch.randn(self.rb_forces[force_indices, :].shape, device=self.device) * obj_mass[force_indices, None] * self.cfg.force_scale
            self.object.permanent_wrench_composer.set_forces_and_torques(
                forces=self.rb_forces.reshape(self.num_envs, 1, 3),
                torques=torch.zeros(self.num_envs, 1, 3, device=self.device),
            )

    def _apply_action(self) -> None:
        """Torque control: torques = p_gain*(target - pos) - d_gain*vel, sent via set_joint_effort_target.
        p_gain/d_gain are hardcoded from cfg (2.0/0.2), NOT read from URDF/USD stiffness/damping."""
        self._refresh_lab()
        if self.cfg.torque_control:
            self.torques = self.p_gain * (self.cur_targets - self.hand_dof_pos) - self.d_gain * self.hand_dof_vel
            self.hand.set_joint_effort_target(self.torques[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        else:
            self.hand.set_joint_position_target(self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    def _get_observations(self) -> dict:
        self._refresh_lab()
        obs = self.compute_observations()
        return {
            "obs":          obs,
            "priv_info":    self.priv_info_buf.clone(),
            "proprio_hist": self.proprio_hist_buf.clone(),
        }

    def _get_rewards(self) -> torch.Tensor:
        """Compute 6-term reward. Total ×0.01 for PPO.
        Angular velocity via quaternion difference, no angle gate (pure axis projection)."""
        object_angvel = axis_angle_from_quat(quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))) / self.step_dt
        # (1) rotate_reward: clip(angvel·Z_axis, -0.5, 0.5) × 2.5 — encourage Z-axis rotation
        rotate_reward = saturate((object_angvel * self.rot_axis).sum(-1), torch.tensor(self.cfg.angvel_clip_min), torch.tensor(self.cfg.angvel_clip_max))
        # (2) object_linvel_penalty: ‖Δpos‖₁/dt × (-0.3) — suppress object translation
        object_linvel_penalty = torch.norm(self.object_pos - self.object_pos_prev, p=1, dim=-1) / self.step_dt
        # (3) pos_diff_penalty: Σ(joint_pos - init_joint_pos)² × (-0.4) — stay near grasp pose
        pos_diff_penalty = ((self.hand_dof_pos[:, self.actuated_dof_indices] - self.init_joint_pos[:, self.actuated_dof_indices]) ** 2).sum(-1)
        # (4) torque_penalty: Σ(self.torques²) × (-0.1) — suppress excessive PD torques (explicit, not PhysX)
        torque_penalty = (self.torques[:, self.actuated_dof_indices] ** 2).sum(-1)
        # (5) work_penalty: (Σ(torques·vel))² × (-0.5) — suppress mechanical power
        work_penalty = ((self.torques[:, self.actuated_dof_indices] * self.hand_dof_vel[:, self.actuated_dof_indices]).sum(-1)) ** 2
        # (6) object_pos_reward: 1/(d+0.001) × 0.003 — stay near initial position
        object_pos_diff = 1.0 / (torch.norm(self.object_pos - self.object_default_pose.clone()[:, :3], dim=-1) + 0.001)

        total_reward = compute_rewards(
            rotate_reward, self.cfg.rotate_reward_scale,
            object_linvel_penalty, self.cfg.object_linvel_penalty_scale,
            pos_diff_penalty, self.cfg.pos_diff_penalty_scale,
            torque_penalty, self.cfg.torque_penalty_scale,
            work_penalty, self.cfg.work_penalty_scale,
            object_pos_diff, self.cfg.object_pos_reward_scale,
        )

        self.extras["rotate_reward"] = rotate_reward.mean()
        self.extras["object_linvel_penalty"] = object_linvel_penalty.mean()
        self.extras["pos_diff_penalty"] = pos_diff_penalty.mean()
        self.extras["torque_penalty"] = torque_penalty.mean()
        self.extras["work_penalty"] = work_penalty.mean()
        self.extras['object_pos_reward'] = (self.cfg.object_pos_reward_scale * object_pos_diff).mean()
        self.extras['angvelX'] = object_angvel[:, 0].mean()
        self.extras['angvelY'] = object_angvel[:, 1].mean()
        self.extras['angvelZ'] = object_angvel[:, 2].mean()
        self.extras['gravity_z'] = self.physics_sim_view.get_gravity()[2]
        self.extras['total_reward'] = total_reward.mean()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Termination: height out of [init_z±2cm] or episode timeout (400 steps @20Hz).
        Gravity curriculum: when reset rate < 0.05%, gravity += 0.05 from -0.05 up to 10."""
        self._refresh_lab()
        height_reset_upper = self.object_pos[:, 2] > self.reset_height_upper
        height_reset_lower = self.object_pos[:, 2] < self.reset_height_lower
        height_reset = height_reset_upper | height_reset_lower
        time_out = self.episode_length_buf >= self.max_episode_length
        self.extras['height_reset_upper'] = height_reset_upper.float().mean()
        self.extras['height_reset_lower'] = height_reset_lower.float().mean()
        self.extras['time_out'] = time_out.float().mean()
        if self.extras['height_reset_upper'] < 5e-4 and self.extras['height_reset_lower'] < 5e-4 and self.cfg.gravity_curriculum and self.common_step_counter > 1000:
            gravity_amp = self.physics_sim_view.get_gravity()
            gravity_amp = torch.sqrt(torch.tensor(gravity_amp[0]**2+gravity_amp[1]**2+gravity_amp[2]**2))
            if gravity_amp < 10: # max gravity set to 10
                new_gravity = carb.Float3(0.0, 0.0, -gravity_amp - 0.05)
                self.physics_sim_view.set_gravity(new_gravity)
                print(f"update gravity: {new_gravity}")
        return height_reset, time_out

    def _rand_pd_scales(self, lower, upper, num_envs, n_dofs):
        rand_scale_s = torch.distributions.Uniform(lower, 1).sample((num_envs, n_dofs)).to(self.device)
        rand_scale_l = torch.distributions.Uniform(1, upper).sample((num_envs, n_dofs)).to(self.device)
        mask_choice = torch.rand((num_envs, n_dofs), device=self.device) > 0.5
        rand_scale = torch.where(mask_choice, rand_scale_s, rand_scale_l)
        return rand_scale

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset hand to grasp pose (from cache or init_joint_pos), object to default state.
        PD gains randomized per-DOF each reset: p_gain × [0.5,2.0], d_gain × [0.5,2.0].
        Height bounds computed dynamically: obj_z ± 2cm window."""
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # pd randomize
        if self.cfg.randomize_pd_gains:
            assert self.cfg.randomize_p_gain_scale_lower <= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_p_gain_scale_upper >= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_d_gain_scale_lower <= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            assert self.cfg.randomize_d_gain_scale_upper >= 1, "pd scale lower bound must be <= 1, upper bound must be >= 1"
            rand_scale = self._rand_pd_scales(self.cfg.randomize_p_gain_scale_lower, self.cfg.randomize_p_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.p_gain[env_ids] = self.cfg.pgain * rand_scale
            rand_scale = self._rand_pd_scales(self.cfg.randomize_d_gain_scale_lower, self.cfg.randomize_d_gain_scale_upper, len(env_ids), self.num_hand_dofs)
            self.d_gain[env_ids] = self.cfg.dgain * rand_scale

        # pose cache
        ndof_cache = self.num_hand_dofs
        if self.saved_grasping_states is not None:
            sampled_idx = torch.randint(0, self.saved_grasping_states.shape[0], (len(env_ids),), device=self.device)
            sampled_pose = self.saved_grasping_states[sampled_idx].clone()
        else:
            sampled_pose = torch.cat([
                self.init_joint_pos.expand(len(env_ids), -1),
                self.object.data.default_root_state[env_ids, :3],
                self.object.data.default_root_state[env_ids, 3:7],
            ], dim=-1)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        if self.cfg.reset_random_quat:
            rotate_center = self.hand.data.default_root_state.clone()[env_ids, :3]
            q_rand = get_random_rotation(env_ids, self.device)
            _, object_default_pos = apply_random_rotation_with_center(object_default_state[:, 3:7], object_default_state[:, 0:3], rotate_center, q_rand)
            self.object_default_pose[env_ids, :3] = object_default_pos.clone()
            object_default_state[:, 3:7], object_default_state[:, 0:3] = apply_random_rotation_with_center(sampled_pose[:, ndof_cache+3:ndof_cache+7], sampled_pose[:, ndof_cache:ndof_cache+3], rotate_center, q_rand)
            object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        else:
            self.object_default_pose[env_ids, :3] = object_default_state[:, :3].clone()
            object_default_state[:, 0:3] = sampled_pose[:, ndof_cache:ndof_cache+3] + self.scene.env_origins[env_ids]
            object_default_state[:, 3:7] = sampled_pose[:, ndof_cache+3:ndof_cache+7]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.object_default_pose[env_ids, 3:7] = object_default_state[:, 3:7]
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = object_default_state[:, 2] - (self.cfg.reset_height_upper - self.cfg.reset_height_lower) / 2
        self.reset_height_upper[env_ids] = object_default_state[:, 2] + (self.cfg.reset_height_upper - self.cfg.reset_height_lower) / 2

        # reset hand
        hand_default_state = self.hand.data.default_root_state.clone()[env_ids]
        if self.cfg.reset_random_quat:
            hand_default_state[:, 3:7], hand_default_state[:, 0:3] = apply_random_rotation_with_center(hand_default_state[:, 3:7], hand_default_state[:, :3], rotate_center, q_rand)
        hand_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        self.hand.write_root_state_to_sim(hand_default_state, env_ids)
        dof_pos = sampled_pose[:, :ndof_cache]
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self._refresh_lab()
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _refresh_lab(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(self.num_envs, self.num_fingertips, 3)
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # visualize coordinate axes for cylinder using VisualizationMarkers
        if getattr(self.cfg, 'debug_show_axes', True) and self._axes_visualizer is not None and self.num_envs > 0:
            try:
                # world poses are already with env origins; add back origins for vis API if needed
                cyl_pos_w = self.object.data.root_pos_w
                cyl_quat_w = self.object.data.root_quat_w
                self._axes_visualizer.visualize(translations=cyl_pos_w, orientations=cyl_quat_w)
            except Exception:
                pass

    def compute_observations(self):
        # contact forces with smoothing + latency
        net_contact_forces_history = torch.cat([self._contact_sensor[id].data.net_forces_w_history[:, :, 0, :].unsqueeze(2) for id in self._contact_body_ids], dim=2)
        norm_contact_forces_history = torch.norm(net_contact_forces_history, dim=-1)
        smooth_contact_forces = norm_contact_forces_history[:, 0, :] * self.cfg.contact_smooth + norm_contact_forces_history[:, 1, :] * (1 - self.cfg.contact_smooth)
        smooth_contact_forces[:, self._contact_body_ids_disable] = 0.0
        if self.cfg.binary_contact:
            binary_contacts = torch.where(smooth_contact_forces > self.cfg.contact_threshold, 1.0, 0.0)
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + binary_contacts * (1 - latency)
            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.cfg.contact_sensor_noise, 0.0, 1.0)
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
        else:
            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.cfg.contact_latency, 1.0, 0.0)
            self.last_contacts = self.last_contacts * latency + smooth_contact_forces * (1 - latency)
            sensed_contacts = self.last_contacts.clone()

        # contact_pos computation retained for future reference (always zeroed: enable_contact_pos=False)
        # not_contact_mask = sensed_contacts < 1.0e-6
        # not_contact_mask[:, self._contact_body_ids_disable] = True
        # contact_mask = ~not_contact_mask
        # contact_pos = torch.cat([self._contact_sensor[id].data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for id in self._contact_body_ids], dim=1)
        # contact_pos = torch.nan_to_num(contact_pos, nan=0.0)
        # contact_pos[contact_mask, :] = transform_between_frames(contact_pos[contact_mask, :] - tactile_frame_pos[contact_mask, :], world_quat[contact_mask, :], tactile_frame_quat[contact_mask, :])
        # contact_pos[not_contact_mask, :] = 0.0
        # contact_pos = contact_pos.reshape(self.num_envs, -1)
        # if not self.cfg.enable_contact_pos:
        #     contact_pos[:] = 0.0

        if not self.cfg.enable_tactile:
            sensed_contacts[:] = 0.0

        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = (torch.rand(self.hand_dof_pos.shape, device=self.device) * 2.0 - 1.0) * self.cfg.joint_noise_scale
        cur_obs_buf = unscale(
            joint_noise_matrix + self.hand_dof_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits
        ).clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat([cur_obs_buf, sensed_contacts.clone().unsqueeze(1)], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        ndof = self.num_hand_dofs
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:ndof] = unscale(
            self.hand_dof_pos[at_reset_env_ids],
            self.hand_dof_lower_limits[at_reset_env_ids],
            self.hand_dof_upper_limits[at_reset_env_ids],
        ).clone().unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof:ndof*2] = self.hand_dof_pos[at_reset_env_ids].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, ndof*2:ndof*2+5] = sensed_contacts[at_reset_env_ids].unsqueeze(1)
        self.at_reset_buf[at_reset_env_ids] = 0
        obs_buf = (self.obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone()

        # Stage2: zero contacts in actor obs, proprio_hist retains real contact history
        if not self.cfg.enable_contact_in_obs:
            obs_single = ndof * 2 + 5
            for f in range(3):
                obs_buf[:, f * obs_single + ndof * 2:f * obs_single + ndof * 2 + 5] = 0.0

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.cfg.prop_hist_len:].clone()
        self.priv_info_buf[:, 0:3] = self.object_pos - self.object_default_pose[:, :3]

        return obs_buf

    def set_friction(self, asset, value, num_envs):
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def set_com(self, asset, value, num_envs):
        coms = asset.root_physx_view.get_coms().clone()
        coms[:, :3] += value
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_coms(coms, env_ids)

    def set_mass(self, asset, value, num_envs):
        env_ids = torch.arange(num_envs, device="cpu")
        asset.root_physx_view.set_masses(value, env_ids)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
def compute_rewards(
    rotate_reward: torch.Tensor, rotate_reward_scale: float,
    object_linvel_penalty: torch.Tensor, object_linvel_penalty_scale: float,
    pos_diff_penalty: torch.Tensor, pos_diff_penalty_scale: float,
    torque_penalty: torch.Tensor, torque_penalty_scale: float,
    work_penalty: torch.Tensor, work_penalty_scale: float,
    object_pos_diff: torch.Tensor, object_pos_reward_scale: float,
):
    reward = rotate_reward * rotate_reward_scale
    reward += object_linvel_penalty * object_linvel_penalty_scale
    reward += pos_diff_penalty * pos_diff_penalty_scale
    reward += torque_penalty * torque_penalty_scale
    reward += work_penalty * work_penalty_scale
    reward += object_pos_diff * object_pos_reward_scale
    return reward

@torch.jit.script
def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.zeros((B, 3, 3), device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

@torch.jit.script
def get_random_rotation(env_ids: torch.Tensor, device: str) -> torch.Tensor:
    N = env_ids.shape[0]

    u1 = torch.rand(N, device=device)
    u2 = torch.rand(N, device=device) * 2.0 * torch.pi
    u3 = torch.rand(N, device=device) * 2.0 * torch.pi
    q1 = torch.sqrt(1.0 - u1) * torch.sin(u2)
    q2 = torch.sqrt(1.0 - u1) * torch.cos(u2)
    q3 = torch.sqrt(u1) * torch.sin(u3)
    q4 = torch.sqrt(u1) * torch.cos(u3)
    q_rand = torch.stack([q4, q1, q2, q3], dim=-1)

    return q_rand

@torch.jit.script
def apply_random_rotation_with_center(
    qs_init: torch.Tensor, pos_init: torch.Tensor, center: torch.Tensor, q_rand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qs_new = quat_mul(q_rand, qs_init)

    R = quat_to_rotmat(q_rand)
    offset = pos_init - center
    new_offset = torch.bmm(R, offset.unsqueeze(-1)).squeeze(-1)
    pos_new = new_offset + center

    return qs_new, pos_new

@torch.jit.script
def rotate_axis_by_quat(axis: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    axis_q = torch.cat([torch.zeros(axis.shape[:-1] + (1,), device=axis.device), axis], dim=-1)
    quat_conj = quat_conjugate(quat)
    rotated_q = quat_mul(quat_mul(quat, axis_q), quat_conj)
    return rotated_q[..., 1:]
