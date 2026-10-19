#!/usr/bin/env python3
"""Offline asset contract check; requires usd-core, no Isaac Sim or GPU."""
import importlib.util
from pathlib import Path
import subprocess

from pxr import Usd, UsdPhysics, UsdUtils

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("description", ROOT / "source/BrainCo_DexHand/BrainCo_DexHand/assets/description.py")
description = importlib.util.module_from_spec(spec)
spec.loader.exec_module(description)


def main():
    revision = subprocess.check_output(["git", "-C", str(description.DESCRIPTION_ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert revision == description.DESCRIPTION_REVISION, (revision, description.DESCRIPTION_REVISION)
    for path, joint_count, body_count, sides in (
        (description.REVO3_RIGHT_USD, 21, 22, ("right",)),
        (description.REVOTRON_SINGLE_RIGHT_USD, 28, 29, ("right",)),
        (description.REVOTRON_BIMANUAL_USD, 59, 60, ("right", "left")),
    ):
        stage = Usd.Stage.Open(path)
        assert stage and stage.GetDefaultPrim()
        unresolved = set(UsdUtils.ComputeAllDependencies(path)[2]) - {"OmniPBR.mdl"}  # supplied by Isaac Sim
        assert not unresolved, f"Unresolved references in {path}: {unresolved}"
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert len(joints) == joint_count and len(bodies) == body_count, (path, len(joints), len(bodies))
        for prim in joints:
            joint = UsdPhysics.RevoluteJoint(prim)
            assert joint.GetLowerLimitAttr().Get() < joint.GetUpperLimitAttr().Get(), str(prim.GetPath())
        for side in sides:
            for name in [f"{side}_palm", *[f"{side}_{finger}_tip_Link" for finger in ("thumb", "index", "middle", "ring", "little")]]:
                body, position, quat = description.fixed_frame(path, name)
                assert body in {p.GetName() for p in bodies}
                assert abs(sum(x*x for x in quat) - 1.0) < 1e-6
                assert len(position) == 3
        print(f"ASSET_PASS {Path(path).name}: {joint_count} joints, {body_count} bodies")


if __name__ == "__main__":
    main()
