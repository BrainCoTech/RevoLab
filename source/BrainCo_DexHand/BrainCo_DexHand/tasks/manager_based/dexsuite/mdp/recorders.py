# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers.recorder_manager import RecorderTerm


class PreStepDataCollectionObservationsRecorder(RecorderTerm):
    """Recorder term that saves the ``data_collection`` observation group each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf["data_collection"]
