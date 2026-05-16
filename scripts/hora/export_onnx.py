#!/usr/bin/env python3
"""Export Stage1/Stage2 checkpoint to ONNX.

Stage1 IO:  obs[B,141] + priv_info[B,8] → action[B,21]
Stage2 IO:  obs[B,141] + proprio_hist[B,30,47] → action[B,21]

Normalization (RunningMeanStd + sa_mean_std) is baked into the ONNX graph via
Stage1ExportWrapper / Stage2ExportWrapper. Output is clamped to [-1,1].

Outputs <name>.onnx and <name>.deploy_meta.yaml (IO contract, joint order, action semantics).
Default obs_dim=141 (no contact_pos), default output path under outputs/hora/revo3_right/onnx/.

Gotcha — obs_dim must match the checkpoint's training config. The default 141 matches
  the current env (3 frames × 47 dims, no contact_pos). Old checkpoints with contact_pos
  (186 dims) need --obs_dim 186 --prop_hist_len 30.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

# Ensure the extension package is importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_PATH = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(EXTENSION_PATH))

from BrainCo_DexHand.algo.hora.models.models import ActorCritic
from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd


DEFAULT_ACTOR_UNITS = [512, 256, 128]
DEFAULT_PRIV_UNITS = [256, 128, 8]
DEFAULT_PRIV_DIM = 8
DEFAULT_OBS_DIM = 141
DEFAULT_ACTION_DIM = 21
DEFAULT_PROP_HIST_LEN = 30
DEFAULT_OBS_PER_STEP = 47

RIGHT_HAND_JOINT_ORDER = [
    "right_index_MPR_joint",
    "right_little_MPR_joint",
    "right_middle_MPR_joint",
    "right_ring_MPR_joint",
    "right_thumb_CMP_joint",
    "right_index_MCP_joint",
    "right_little_MCP_joint",
    "right_middle_MCP_joint",
    "right_ring_MCP_joint",
    "right_thumb_CMR_joint",
    "right_index_PIP_joint",
    "right_little_PIP_joint",
    "right_middle_PIP_joint",
    "right_ring_PIP_joint",
    "right_thumb_MCP_joint",
    "right_index_DIP_joint",
    "right_little_DIP_joint",
    "right_middle_DIP_joint",
    "right_ring_DIP_joint",
    "right_thumb_PIP_joint",
    "right_thumb_DIP_joint",
]


def _find_config_for_checkpoint(ckpt_path: Path) -> Path | None:
    # expected: run_dir/stage2_nn/model_best.ckpt or run_dir/stage1_nn/best.pth
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name.endswith("_nn") else ckpt_path.parent
    candidates = sorted(run_dir.glob("config_*.yaml"))
    return candidates[-1] if candidates else None


def _find_run_dir_for_checkpoint(ckpt_path: Path) -> Path:
    # expected: outputs/hora/revo3_right/<run_name>/stage{1,2}_nn/*.pth|*.ckpt
    if ckpt_path.parent.name.endswith("_nn"):
        return ckpt_path.parent.parent
    return ckpt_path.parent


def _load_config(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    return OmegaConf.load(str(path))


def _get_cfg_value(cfg: Any | None, key_path: str, default: Any) -> Any:
    if cfg is None:
        return default
    node: Any = cfg
    for key in key_path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
            continue
        if hasattr(node, key):
            node = getattr(node, key)
            continue
        return default
    return node


def _build_net_config(cfg: Any | None, stage: str, obs_dim: int, actions_num: int) -> dict[str, Any]:
    actor_units = list(_get_cfg_value(cfg, "train.network.mlp.units", DEFAULT_ACTOR_UNITS))
    priv_units = list(_get_cfg_value(cfg, "train.network.priv_mlp.units", DEFAULT_PRIV_UNITS))
    priv_dim = int(_get_cfg_value(cfg, "train.ppo.priv_info_dim", DEFAULT_PRIV_DIM))
    priv_info_cfg = bool(_get_cfg_value(cfg, "train.ppo.priv_info", True))

    if stage == "stage2":
        priv_info = True
        proprio_adapt = True
    else:
        priv_info = priv_info_cfg
        proprio_adapt = False

    return {
        "actor_units": actor_units,
        "priv_mlp_units": priv_units,
        "actions_num": actions_num,
        "input_shape": (obs_dim,),
        "priv_info": priv_info,
        "proprio_adapt": proprio_adapt,
        "priv_info_dim": priv_dim,
        "obs_per_step": obs_dim // 3,
    }


class Stage2ExportWrapper(torch.nn.Module):
    def __init__(self, model: ActorCritic, rms_obs: RunningMeanStd, rms_hist: RunningMeanStd):
        super().__init__()
        self.model = model
        self.rms_obs = rms_obs
        self.rms_hist = rms_hist

    def forward(self, obs: torch.Tensor, proprio_hist: torch.Tensor) -> torch.Tensor:
        obs_n = self.rms_obs(obs)
        hist_n = self.rms_hist(proprio_hist)
        mu = self.model.act_inference({"obs": obs_n, "proprio_hist": hist_n})
        return torch.clamp(mu, -1.0, 1.0)


class Stage1ExportWrapper(torch.nn.Module):
    def __init__(self, model: ActorCritic, rms_obs: RunningMeanStd):
        super().__init__()
        self.model = model
        self.rms_obs = rms_obs

    def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
        obs_n = self.rms_obs(obs)
        mu = self.model.act_inference({"obs": obs_n, "priv_info": priv_info})
        return torch.clamp(mu, -1.0, 1.0)


def _save_deploy_meta(
    meta_path: Path,
    *,
    stage: str,
    checkpoint: Path,
    onnx_path: Path,
    cfg_path: Path | None,
    obs_dim: int,
    action_dim: int,
    prop_hist_len: int,
    obs_per_step: int,
    dynamic_batch: bool,
    normalize_baked_in: bool,
    policy_rate: float,
    chunk_size: int,
    n_action_steps: int,
    runtime_reference: dict[str, Any],
) -> None:
    meta = {
        "export": {
            "stage": stage,
            "config_yaml": str(cfg_path) if cfg_path is not None else "",
        },
        "io_contract": {
            "inputs": (
                [
                    {"name": "obs", "shape": ["B", obs_dim], "dtype": "float32"},
                    {"name": "proprio_hist", "shape": ["B", prop_hist_len, obs_per_step], "dtype": "float32"},
                ]
                if stage == "stage2"
                else [
                    {"name": "obs", "shape": ["B", obs_dim], "dtype": "float32"},
                    {"name": "priv_info", "shape": ["B", DEFAULT_PRIV_DIM], "dtype": "float32"},
                ]
            ),
            "outputs": [{"name": "action", "shape": ["B", action_dim], "dtype": "float32"}],
            "action_semantics": "delta",
            "action_formula": "cur_targets = prev_targets + (1/24) * action, then clamp to joint limits",
            "action_clip": [-1.0, 1.0],
            "policy_rate_hz": float(policy_rate),
            "chunk_size": int(chunk_size),
            "n_action_steps": int(n_action_steps),
            "joint_order_right_hand": RIGHT_HAND_JOINT_ORDER,
            "dynamic_batch": dynamic_batch,
        },
        "normalization": {
            "baked_in_onnx": normalize_baked_in,
            "source": "checkpoint running_mean_std (+ sa_mean_std for stage2)",
        },
        "runtime_reference": runtime_reference,
        "deploy_keep_from_config": [
            "train.network.mlp.units",
            "train.network.priv_mlp.units",
            "train.ppo.priv_info",
            "train.ppo.priv_info_dim",
            "env_runtime.randomize_scale_list",
            "env_runtime.grasp_cache_file",
            "env_runtime.usd_path",
            "env_runtime.reward.*",
            "env_runtime.termination.*",
            "env_runtime.dr.*",
        ],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(meta)))


def _shape_list_from_state(state: dict[str, Any], key: str) -> list[int] | None:
    tensor = state.get(key)
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return [int(x) for x in tuple(tensor.shape)]


def _resolve_scale_keys_from_config(cfg: Any | None) -> list[float]:
    scale_list = _get_cfg_value(cfg, "env_runtime.randomize_scale_list", None)
    if scale_list is not None:
        parsed = [float(s) for s in scale_list]
        if len(parsed) > 0:
            return parsed
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage1/Stage2 checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (stage1) or .ckpt (stage2).")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output ONNX path. If empty, use policy_MMDDHH.onnx.",
    )
    parser.add_argument("--stage", type=str, default="stage2", choices=["stage1", "stage2"])
    parser.add_argument("--config", type=str, default="", help="Optional config_*.yaml. Auto-resolved if empty.")
    parser.add_argument("--obs_dim", type=int, default=DEFAULT_OBS_DIM)
    parser.add_argument("--action_dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--prop_hist_len", type=int, default=DEFAULT_PROP_HIST_LEN)
    parser.add_argument("--policy_rate", type=float, default=20.0, help="Policy control frequency in Hz.")
    parser.add_argument("--chunk_size", type=int, default=1, help="Chunk size for action playback.")
    parser.add_argument("--n_action_steps", type=int, default=1, help="Number of action steps per chunk.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no_dynamic_batch", action="store_true", help="Disable dynamic batch axis.")
    parser.add_argument(
        "--meta_output",
        type=str,
        default="",
        help="Optional deploy meta yaml path. Default: <output_without_ext>.deploy_meta.yaml",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_dir = _find_run_dir_for_checkpoint(ckpt_path)
    onnx_dir = Path("outputs") / "hora" / "revo3_right" / "onnx"
    output_name = args.output.strip() if isinstance(args.output, str) else ""
    if not output_name:
        # Default: outputs/hora/revo3_right/onnx/policy_MMDDHH.onnx
        out_path = (onnx_dir / f"policy_{datetime.datetime.now().strftime('%m%d%H')}.onnx").resolve()
    else:
        out_arg_path = Path(output_name).expanduser()
        if out_arg_path.is_absolute() or out_arg_path.parent != Path("."):
            out_path = out_arg_path.resolve()
        else:
            # If only a filename is given, still place it under outputs/hora/revo3_right/onnx/
            out_path = (onnx_dir / out_arg_path.name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config).expanduser().resolve() if args.config else _find_config_for_checkpoint(ckpt_path)
    cfg = _load_config(cfg_path)
    if cfg_path is None:
        print("[WARN] config_*.yaml not found near checkpoint; using script defaults.")
    else:
        print(f"[INFO] Using config: {cfg_path}")

    obs_dim = int(args.obs_dim)
    actions_num = int(args.action_dim)
    prop_hist_len = int(args.prop_hist_len)
    scale_keys = _resolve_scale_keys_from_config(cfg)
    obs_per_step = obs_dim // 3
    if args.stage == "stage2" and obs_per_step != DEFAULT_OBS_PER_STEP:
        print(
            f"[WARN] obs_per_step={obs_per_step} (from obs_dim={obs_dim}) "
            f"!= expected Stage2 default {DEFAULT_OBS_PER_STEP}.",
            flush=True,
        )

    net_config = _build_net_config(cfg, args.stage, obs_dim=obs_dim, actions_num=actions_num)
    model = ActorCritic(net_config).cpu().eval()
    rms_obs = RunningMeanStd((obs_dim,)).cpu().eval()

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    if "running_mean_std" in checkpoint:
        rms_obs.load_state_dict(checkpoint["running_mean_std"])
    else:
        print("[WARN] running_mean_std missing in checkpoint; obs normalization will use default stats.")

    rms_obs_shape = None
    if "running_mean_std" in checkpoint and isinstance(checkpoint["running_mean_std"], dict):
        rms_obs_shape = _shape_list_from_state(checkpoint["running_mean_std"], "running_mean")

    dynamic_batch = not args.no_dynamic_batch
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {"action": {0: "B"}}

    if args.stage == "stage2":
        rms_hist = RunningMeanStd((prop_hist_len, obs_per_step)).cpu().eval()
        rms_hist_shape = None
        if "sa_mean_std" in checkpoint:
            rms_hist.load_state_dict(checkpoint["sa_mean_std"])
            if isinstance(checkpoint["sa_mean_std"], dict):
                rms_hist_shape = _shape_list_from_state(checkpoint["sa_mean_std"], "running_mean")
        else:
            print("[WARN] sa_mean_std missing in checkpoint; proprio_hist normalization will use default stats.")

        expected_obs = [obs_dim]
        expected_hist = [prop_hist_len, obs_per_step]
        if rms_obs_shape is not None and rms_obs_shape != expected_obs:
            print(f"[WARN] running_mean_std shape mismatch: ckpt={rms_obs_shape}, expected={expected_obs}")
        if rms_hist_shape is not None and rms_hist_shape != expected_hist:
            print(f"[WARN] sa_mean_std shape mismatch: ckpt={rms_hist_shape}, expected={expected_hist}")
        print(
            f"[INFO] Stage2 reference -> RMS(obs)={expected_obs}, RMS(hist)={expected_hist}, "
            f"joints={len(RIGHT_HAND_JOINT_ORDER)}, scale_keys={scale_keys if scale_keys else 'unknown'}",
            flush=True,
        )

        wrapper = Stage2ExportWrapper(model, rms_obs, rms_hist).eval()
        obs = torch.zeros((1, obs_dim), dtype=torch.float32)
        proprio_hist = torch.zeros((1, prop_hist_len, obs_per_step), dtype=torch.float32)
        input_names = ["obs", "proprio_hist"]
        output_names = ["action"]
        if dynamic_batch:
            dynamic_axes = {
                "obs": {0: "B"},
                "proprio_hist": {0: "B"},
                "action": {0: "B"},
            }
        torch.onnx.export(
            wrapper,
            (obs, proprio_hist),
            str(out_path),
            opset_version=args.opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    else:
        wrapper = Stage1ExportWrapper(model, rms_obs).eval()
        obs = torch.zeros((1, obs_dim), dtype=torch.float32)
        priv_info_dim = int(net_config["priv_info_dim"])
        priv_info = torch.zeros((1, priv_info_dim), dtype=torch.float32)
        input_names = ["obs", "priv_info"]
        output_names = ["action"]
        if dynamic_batch:
            dynamic_axes = {
                "obs": {0: "B"},
                "priv_info": {0: "B"},
                "action": {0: "B"},
            }
        torch.onnx.export(
            wrapper,
            (obs, priv_info),
            str(out_path),
            opset_version=args.opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    meta_path = (
        Path(args.meta_output).expanduser().resolve()
        if args.meta_output
        else out_path.with_suffix(".deploy_meta.yaml")
    )
    _save_deploy_meta(
        meta_path,
        stage=args.stage,
        checkpoint=ckpt_path,
        onnx_path=out_path,
        cfg_path=cfg_path,
        obs_dim=obs_dim,
        action_dim=actions_num,
        prop_hist_len=prop_hist_len,
        obs_per_step=obs_per_step,
        dynamic_batch=dynamic_batch,
        normalize_baked_in=True,
        policy_rate=float(args.policy_rate),
        chunk_size=int(args.chunk_size),
        n_action_steps=int(args.n_action_steps),
        runtime_reference={
            "scale_keys": scale_keys,
            "running_mean_std_obs_shape": rms_obs_shape if rms_obs_shape is not None else [obs_dim],
            "running_mean_std_hist_shape": (
                rms_hist_shape if args.stage == "stage2" and rms_hist_shape is not None else [prop_hist_len, obs_per_step]
            ),
            "joint_order_source": "Isaac Lab runtime hand.data.joint_names",
            "joint_order_right_hand": RIGHT_HAND_JOINT_ORDER,
        },
    )

    print(f"[OK] Exported ONNX: {out_path}")
    print(f"[OK] Export metadata: {meta_path}")


if __name__ == "__main__":
    main()
