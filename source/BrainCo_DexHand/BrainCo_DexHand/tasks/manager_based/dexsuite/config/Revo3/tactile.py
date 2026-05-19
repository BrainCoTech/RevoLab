# Copyright (c) 2026, BrainCo.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tactile sensor constants and config helpers for the right Revo3 hand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from isaaclab.sensors import ContactSensorCfg

try:
    from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor as _VisuoTactileSensor
except ImportError:
    _VisuoTactileSensor = None


TACTILE_FINGER_ORDER = ("little", "ring", "middle", "index", "thumb")

TACTILE_DIP_BODIES = tuple(f"right_{finger}_DIP_Link" for finger in TACTILE_FINGER_ORDER)
TACTILE_TIP_BODIES = tuple(f"right_{finger}_tip_Link" for finger in TACTILE_FINGER_ORDER)

TACTILE_FORCE_SENSOR_NAMES = tuple(f"{finger}_tactile_force" for finger in TACTILE_FINGER_ORDER)
TACTILE_VIS_SENSOR_NAMES = tuple(f"{finger}_tactile_sensor" for finger in TACTILE_FINGER_ORDER)

TACTILE_USD_PATH = (
    Path(__file__).resolve().parents[8] / "assets" / "usd" / "dexsuite" / "Tianji_Revo3_Right_tactile.usda"
)
TACTILE_CUBE_USD_PATH = (
    Path(__file__).resolve().parents[8] / "assets" / "usd" / "dexsuite" / "tactile_cube_sdf.usda"
)


if _VisuoTactileSensor is not None:

    class Revo3VisuoTactileSensor(_VisuoTactileSensor):
        """TacSL sensor variant that initializes the no-contact camera baseline automatically."""

        def _initialize_camera_tactile(self):
            super()._initialize_camera_tactile()
            self.get_initial_render()

else:

    class Revo3VisuoTactileSensor:
        """Placeholder used only when TacSL contrib sensors are not importable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TacSL tactile camera support requires "
                "`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor`."
            )


@dataclass(frozen=True)
class TactileCameraSettings:
    """Camera settings shared by the five fingertip TacSL sensors."""

    height: int = 320
    width: int = 240
    update_period: float = 0.0


def make_tactile_force_sensor_cfgs() -> dict[str, ContactSensorCfg]:
    """Create one ContactSensorCfg per fingertip DIP link."""

    return {
        sensor_name: ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        for sensor_name, body_name in zip(TACTILE_FORCE_SENSOR_NAMES, TACTILE_DIP_BODIES)
    }


def _load_tacsl_dependencies():
    """Load TacSL classes only when a tactile-camera env is constructed."""

    try:
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab_assets.sensors import GELSIGHT_R15_CFG
        from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
    except ImportError as exc:
        raise ImportError(
            "TacSL tactile camera support requires Isaac Lab contrib sensors. "
            "Install/enable the Isaac Lab extension that provides "
            "`isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensorCfg` and "
            "`isaaclab_assets.sensors.GELSIGHT_R15_CFG`, plus `isaaclab.sensors.TiledCameraCfg`."
        ) from exc

    return GELSIGHT_R15_CFG, TiledCameraCfg, VisuoTactileSensorCfg, Revo3VisuoTactileSensor


def make_tacsl_sensor_cfgs(
    *,
    camera_settings: TactileCameraSettings = TactileCameraSettings(),
    enable_rgb: bool = False,
    enable_force_field: bool = False,
    tactile_array_size: tuple[int, int] = (16, 16),
):
    """Create one VisuoTactileSensorCfg per fingertip.

    The initial training path uses depth plus ContactSensor net forces. TacSL force fields
    stay disabled until the object SDF collision setup is verified.
    """

    gelsight_cfg, tiled_camera_cfg, visuo_tactile_sensor_cfg, sensor_class = _load_tacsl_dependencies()
    sensor_cfgs = {}
    for finger, sensor_name, tip_body in zip(TACTILE_FINGER_ORDER, TACTILE_VIS_SENSOR_NAMES, TACTILE_TIP_BODIES):
        elastomer_path = f"{{ENV_REGEX_NS}}/Robot/{tip_body}/tactile_elastomer"
        sensor_cfgs[sensor_name] = visuo_tactile_sensor_cfg(
            class_type=sensor_class,
            prim_path=f"{elastomer_path}/tactile_sensor",
            history_length=0,
            render_cfg=gelsight_cfg,
            enable_camera_tactile=True,
            enable_force_field=enable_force_field,
            tactile_array_size=tactile_array_size,
            tactile_margin=0.003,
            contact_object_prim_path_expr="{ENV_REGEX_NS}/Object",
            normal_contact_stiffness=1.0,
            friction_coefficient=2.0,
            tangential_stiffness=0.1,
            camera_cfg=tiled_camera_cfg(
                prim_path=f"{elastomer_path}/cam",
                update_period=camera_settings.update_period,
                height=camera_settings.height,
                width=camera_settings.width,
                data_types=["distance_to_image_plane"],
                spawn=None,
            ),
        )

    return sensor_cfgs
