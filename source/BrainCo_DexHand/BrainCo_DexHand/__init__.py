# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BrainCo Isaac Lab extension package."""

# Register Gym environments when Isaac Lab is available. This keeps lightweight
# submodules, such as MuJoCo utilities, importable in non-Isaac runtimes.
try:
    from .tasks import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass

# Register UI extensions when their dependencies are available.
try:
    from .ui_extension_example import *  # noqa: F401,F403
except ModuleNotFoundError:
    pass
