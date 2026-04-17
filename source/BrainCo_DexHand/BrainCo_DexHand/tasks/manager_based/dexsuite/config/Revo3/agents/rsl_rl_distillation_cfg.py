# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class DexsuiteRevo3DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner configuration for Revo3 dexsuite tasks."""

    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 250
    experiment_name = "dexsuite_tianji"

    obs_groups = {
        "policy": ["policy", "student_proprio", "perception"],
        "teacher": ["policy", "proprio", "perception"],
    }

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=True,
        student_hidden_dims=[256, 256, 128],
        teacher_obs_normalization=True,
        teacher_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=4,
        max_grad_norm=1.0,
    )
