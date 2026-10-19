"""Fixed palm/tip frames after upstream fixed-joint merging."""

import torch
from isaaclab.utils.math import quat_apply, quat_mul

from .description import fixed_frame


def frame_state_w(robot, frame_name: str) -> torch.Tensor:
    """World pose and twist [xyz, wxyz, linear xyz, angular xyz] at the frame origin."""
    if not hasattr(robot, "_description_frames"):
        robot._description_frames = {}
    if frame_name not in robot._description_frames:
        body_name, pos, quat = fixed_frame(robot.cfg.spawn.usd_path, frame_name)
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Frame {frame_name} requires exactly one body {body_name}")
        robot._description_frames[frame_name] = (
            body_ids[0], torch.tensor(pos, device=robot.device, dtype=torch.float32),
            torch.tensor(quat, device=robot.device, dtype=torch.float32),
        )
    index, pos, quat = robot._description_frames[frame_name]
    body_quat = robot.data.body_link_quat_w[:, index]
    offset = quat_apply(body_quat, pos.expand(robot.num_instances, -1))
    angular = robot.data.body_link_ang_vel_w[:, index]
    return torch.cat((
        robot.data.body_link_pos_w[:, index] + offset,
        quat_mul(body_quat, quat.expand(robot.num_instances, -1)),
        robot.data.body_link_lin_vel_w[:, index] + torch.cross(angular, offset, dim=-1),
        angular,
    ), dim=-1)


def frames_state_w(robot, frame_names) -> torch.Tensor:
    return torch.stack([frame_state_w(robot, name) for name in frame_names], dim=1)
