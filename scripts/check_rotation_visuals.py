#!/usr/bin/env python3
"""Compare plain and colored HORA objects, including cloning and GPU contact dynamics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "BrainCo_DexHand"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from pxr import Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from BrainCo_DexHand.tasks.direct.hora_rotation.assets import BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG


def physics_attributes(root):
    """Ignore visual-only additions; compare all authored physics schemas and values."""
    result = {}
    for prim in Usd.PrimRange(root):
        schemas = [str(s) for s in prim.GetAppliedSchemas() if "Physics" in str(s) or "Physx" in str(s)]
        if not schemas:
            continue
        path = str(prim.GetPath().MakeRelativePath(root.GetPath()))
        attrs = {a.GetName(): str(a.Get()) for a in prim.GetAuthoredAttributes()
                 if a.GetName().startswith(("physics:", "physx")) or a.GetName() in ("radius", "height", "axis")}
        result[path] = (str(prim.GetTypeName()), schemas, attrs)
    return result


try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
    floor = sim_utils.CuboidCfg(size=(10, 10, 0.1), collision_props=sim_utils.CollisionPropertiesCfg())
    floor.func("/World/floor", floor, translation=(0, 0, -0.05))
    for i in range(2):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(0, i * 2, 0))
    pairs = []
    for name, cfg, plain_spawn in [("ball", BALL_OBJECT_CFG, sim_utils.spawn_sphere),
                                    ("cylinder", CYLINDER_OBJECT_CFG, sim_utils.spawn_cylinder)]:
        colored_cfg = cfg.copy()
        colored_cfg.prim_path = f"/World/env_.*/{name}_colored"
        colored_cfg.init_state.pos = (0.3, 0 if name == "ball" else 0.5, 0.2)
        plain_cfg = colored_cfg.copy()
        plain_cfg.prim_path = f"/World/env_.*/{name}_plain"
        plain_cfg.spawn.func = plain_spawn
        colored, plain = RigidObject(colored_cfg), RigidObject(plain_cfg)
        for i in range(2):
            c = sim.stage.GetPrimAtPath(f"/World/env_{i}/{name}_colored")
            p = sim.stage.GetPrimAtPath(f"/World/env_{i}/{name}_plain")
            assert physics_attributes(c) == physics_attributes(p)
            assert sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(c)) == 1
            mesh = UsdGeom.Mesh(c.GetChild("geometry").GetChild("color_blocks"))
            assert mesh and len(set(map(tuple, mesh.GetDisplayColorPrimvar().Get()))) == 6
            assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
            # Compare at identical world coordinates to avoid position-dependent contact roundoff.
            # Only this overlapping test pair is filtered; both still collide with the same floor.
            UsdPhysics.FilteredPairsAPI.Apply(c).CreateFilteredPairsRel().SetTargets([p.GetPath()])
        pairs.append((name, colored, plain))
    sim.reset()
    for _, colored, plain in pairs:
        torch.testing.assert_close(colored.root_physx_view.get_masses(), plain.root_physx_view.get_masses(), rtol=0, atol=0)
        torch.testing.assert_close(colored.root_physx_view.get_inertias(), plain.root_physx_view.get_inertias(), rtol=0, atol=0)
        torch.testing.assert_close(colored.root_physx_view.get_coms(), plain.root_physx_view.get_coms(), rtol=0, atol=0)
        torch.testing.assert_close(colored.root_physx_view.get_material_properties(), plain.root_physx_view.get_material_properties(), rtol=0, atol=0)
        assert colored.root_physx_view.max_shapes == plain.root_physx_view.max_shapes == 1
    for _ in range(120):
        sim.step(render=False)
        for _, colored, plain in pairs:
            colored.update(sim.get_physics_dt())
            plain.update(sim.get_physics_dt())
            difference = colored.data.root_state_w - plain.data.root_state_w
            torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=1e-5)
    print("ROTATION_VISUALS_PASS: two clones per shape; physics attributes, shape count, mass/inertia and 120 contact steps match", flush=True)
except BaseException:
    import os
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close(wait_for_replicator=False)
