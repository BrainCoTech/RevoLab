"""Record and check the physical and tensor contract beside a training run."""
import json
import hashlib
import subprocess
from pathlib import Path

from .assets.description import DESCRIPTION_REVISION, DESCRIPTION_ROOT


def asset_contract(env, task_id: str) -> dict:
    env = env.unwrapped
    robot = env.hand if hasattr(env, "hand") else env.scene["robot"]
    if hasattr(env, "action_manager"):
        actions = {name: list(env.action_manager.get_term(name)._joint_names)
                   for name in env.action_manager.active_terms}
    else:
        actions = {"joints": [robot.joint_names[i] for i in env.actuated_dof_indices]}
    revision = subprocess.check_output(["git", "-C", str(DESCRIPTION_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if revision != DESCRIPTION_REVISION:
        raise ValueError("Robot submodule differs from the pinned asset revision")
    limits = robot.data.joint_pos_limits[0].cpu().tolist()
    if hasattr(env, "hand_dof_lower_limits"):
        control_limits = list(zip(env.hand_dof_lower_limits[0].cpu().tolist(), env.hand_dof_upper_limits[0].cpu().tolist()))
        control_limits = [list(pair) for pair in control_limits]
    else:
        control_limits = limits
    contract = {
        "task_id": task_id.replace("-Play-v1", "-v1"),
        "asset_revision": revision,
        "asset_sha256": hashlib.sha256(Path(robot.cfg.spawn.usd_path).read_bytes()).hexdigest(),
        "asset_file": str(Path(robot.cfg.spawn.usd_path).relative_to(DESCRIPTION_ROOT)),
        "joint_names": list(robot.joint_names),
        "joint_limits_rad": limits,
        "control_joint_limits_rad": control_limits,
        "action_joints": actions,
        "observation_space": str(env.single_observation_space),
        "action_space": str(env.single_action_space),
    }
    if "Handover" in task_id or "Lift" in task_id:
        # Identical USD/joint spaces do not make checkpoints from a wrong reset pose compatible.
        initial = robot.cfg.init_state
        contract["robot_initial_state"] = {
            "position": list(initial.pos), "rotation": list(initial.rot),
            "joint_positions": dict(initial.joint_pos),
        }
    return contract


def write_asset_contract(env, task_id: str, run_dir: str) -> None:
    path = Path(run_dir) / "params" / "asset_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asset_contract(env, task_id), indent=2) + "\n")


def read_asset_contract(checkpoint: str) -> dict:
    path = Path(checkpoint).resolve()
    for parent in (path.parent, path.parent.parent):
        contract = parent / "params" / "asset_contract.json"
        if contract.is_file():
            return json.loads(contract.read_text())
    raise ValueError(f"Missing params/asset_contract.json for {checkpoint}; v1 requires a new model and its run metadata")


def validate_asset_contract(env, task_id: str, checkpoint: str) -> None:
    actual = read_asset_contract(checkpoint)
    expected = asset_contract(env, task_id)
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f"Checkpoint physical/tensor contract mismatch: {', '.join(mismatches)}")
