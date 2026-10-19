"""Pinned robot descriptions. Task objects and simulation settings live in RevoLab."""

from functools import lru_cache
from pathlib import Path

DESCRIPTION_REVISION = "6db7e2ed1d864ee0a3ab6ff62d43a303787137f6"
REPO_ROOT = Path(__file__).resolve().parents[4]
DESCRIPTION_ROOT = REPO_ROOT / "third_party" / "brainco-description"


def description_path(relative_path: str) -> str:
    path = DESCRIPTION_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing robot asset: {path}. Run scripts/setup_assets.sh.")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Unresolved Git LFS asset: {path}. Run scripts/setup_assets.sh.")
    return str(path)


REVO3_RIGHT_USD = description_path("revo3_system/usd/revo3_right.usd")
REVOTRON_SINGLE_RIGHT_USD = description_path("revotron_system/usd/revotron_single_right_revo3.usd")
REVOTRON_BIMANUAL_USD = description_path("revotron_system/usd/revotron_bimanual_revo3.usd")
LOCKED_REVOTRON_JOINTS = ("waist_pitch_Joint", "neck_yaw_Joint", "head_pitch_Joint")


@lru_cache(maxsize=None)
def fixed_frame(usd_path: str, frame_name: str):
    """Return the rigid ancestor and its fixed transform to a named USD frame."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    matches = [prim for prim in stage.Traverse() if prim.GetName() == frame_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one frame {frame_name!r} in {usd_path}, found {len(matches)}")
    frame = matches[0]
    body = frame
    while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
        body = body.GetParent()
    if not body:
        raise ValueError(f"Frame {frame_name!r} has no rigid-body ancestor")
    cache = UsdGeom.XformCache()
    transform = Gf.Transform(cache.GetLocalToWorldTransform(frame) * cache.GetLocalToWorldTransform(body).GetInverse())
    quat = transform.GetRotation().GetQuat()
    return body.GetName(), tuple(transform.GetTranslation()), (quat.GetReal(), *quat.GetImaginary())


def spawn_locked_revotron(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Lock the unused torso/head DOFs in the scene layer, leaving upstream files intact."""
    from pxr import Usd, UsdPhysics
    from isaaclab.sim import spawn_from_usd, find_matching_prims

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    # spawn_from_usd clones first; author the task constraint on every resulting root.
    for instance in find_matching_prims(prim_path):
        found = set()
        for prim in Usd.PrimRange(instance):
            if prim.GetName() in LOCKED_REVOTRON_JOINTS:
                # A fixed joint has no DOF and cannot drift under arm reaction forces.
                prim.SetTypeName("PhysicsFixedJoint")
                found.add(prim.GetName())
        if found != set(LOCKED_REVOTRON_JOINTS):
            raise ValueError(f"Missing Revotron joints to lock: {set(LOCKED_REVOTRON_JOINTS) - found}")
    return root


def spawn_lift_revotron(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Exclude the overlapping housings of the two co-located wrist joints."""
    from pxr import UsdPhysics
    from isaaclab.sim import spawn_from_usd, find_matching_prims

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for instance in find_matching_prims(prim_path):
        forearm = instance.GetChild("Link5_R")
        wrist = instance.GetChild("Link7_R")
        if not forearm or not wrist:
            raise ValueError("Lift asset is missing the Link5_R/Link7_R wrist pair")
        # Joint6_R and Joint7_R share an origin. Their housing collision meshes
        # overlap in the assembled wrist and generate a large spurious impulse.
        # Keep all other self collisions and all robot/object collisions enabled.
        UsdPhysics.FilteredPairsAPI.Apply(forearm).CreateFilteredPairsRel().AddTarget(wrist.GetPath())
    return root
