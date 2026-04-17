# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.recorders.recorders_cfg import PreStepActionsRecorderCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders


@configclass
class PreStepDataCollectionObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for recording the data_collection observation group."""

    class_type: type[RecorderTerm] = recorders.PreStepDataCollectionObservationsRecorder


@configclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder manager that keeps only action labels and visual-policy inputs."""

    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_data_collection_observations = PreStepDataCollectionObservationsRecorderCfg()
