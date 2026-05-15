from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from revo3_deploy.input_builder import Stage2InputBuilder
from revo3_deploy.robot_profile import JOINT_DIM, Revo3Profile


class Revo3PolicyRunner:
    def __init__(
        self,
        onnx_path: str | Path,
        policy_path: str | Path,
        profile: Revo3Profile,
        use_gpu: bool = False,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.policy_path = Path(policy_path)
        self.policy_cfg = self._load_policy_cfg(self.policy_path)
        self.policy_contract = self._validate_policy_contract(self.policy_cfg)

        policy_order = tuple(self.policy_contract["joint_order_right_hand"])
        if policy_order != profile.policy_joint_order:
            raise ValueError("Policy contract joint order differs from robot profile.")

        self.profile = profile
        self.builder = Stage2InputBuilder(
            joint_lower=profile.joint_lower_policy,
            joint_upper=profile.joint_upper_policy,
            joint_names=profile.policy_joint_order,
            action_scale=profile.action_scale,
        )
        self.initialized = False

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_names = [meta.name for meta in self.session.get_inputs()]
        self.output_name = self._resolve_output_name()
        self.input_aliases = self._build_input_aliases()
        self._validate_onnx_io_contract()

    @property
    def rate_hz(self) -> float:
        return float(self.policy_contract.get("policy_rate_hz") or self.profile.default_rate_hz)

    def step(self, measured_policy_pos_rad: np.ndarray) -> np.ndarray:
        if not self.initialized:
            inputs = self.builder.reset(measured_policy_pos_rad)
            self.initialized = True
        else:
            inputs = self.builder.observe(measured_policy_pos_rad)
        action = self.session.run([self.output_name], self._build_ort_feed(inputs))[0][0]
        return self.builder.action_to_target(action)

    @staticmethod
    def _load_policy_cfg(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_policy_contract(self, policy_cfg: dict) -> dict:
        contract = policy_cfg.get("io_contract")
        if not isinstance(contract, dict):
            raise ValueError("policy.yaml must contain io_contract.")
        for key in ("inputs", "outputs", "joint_order_right_hand"):
            if key not in contract:
                raise ValueError(f"policy.yaml io_contract missing {key}.")
        joint_order = list(contract["joint_order_right_hand"])
        if len(joint_order) != JOINT_DIM or len(set(joint_order)) != JOINT_DIM:
            raise ValueError("joint_order_right_hand must contain 21 unique joints.")
        if str(contract.get("action_semantics", "delta")) != "delta":
            raise ValueError("Only delta action_semantics is supported.")
        return contract

    def _build_ort_feed(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {}
        for input_name in self.input_names:
            builder_key = self.input_aliases.get(input_name) or self._policy_input_to_builder_key(input_name)
            if builder_key is None or builder_key not in inputs:
                raise RuntimeError(f"Unsupported ONNX input name: {input_name}")
            feed[input_name] = inputs[builder_key]
        return feed

    def _build_input_aliases(self) -> dict[str, str]:
        aliases = {"obs": "obs", "proprio_hist": "proprio_hist"}
        for cfg in self.policy_contract.get("inputs", []) or []:
            name = str(cfg.get("name", ""))
            local_name = self._policy_input_to_builder_key(name)
            if local_name:
                aliases[name] = local_name
        return aliases

    def _resolve_output_name(self) -> str:
        ort_output_names = [meta.name for meta in self.session.get_outputs()]
        configured = [
            str(cfg.get("name"))
            for cfg in self.policy_contract.get("outputs", []) or []
            if cfg.get("name")
        ]
        for candidate in configured + ["action"]:
            if candidate in ort_output_names:
                return candidate
        if len(ort_output_names) == 1:
            return ort_output_names[0]
        raise RuntimeError(f"Could not resolve action output from {ort_output_names}.")

    def _validate_onnx_io_contract(self) -> None:
        expected_inputs = [str(cfg.get("name")) for cfg in self.policy_contract.get("inputs", [])]
        expected_outputs = [str(cfg.get("name")) for cfg in self.policy_contract.get("outputs", [])]
        actual_outputs = [meta.name for meta in self.session.get_outputs()]
        if expected_inputs != self.input_names:
            raise ValueError(f"ONNX inputs {self.input_names} do not match policy.yaml {expected_inputs}.")
        if expected_outputs != actual_outputs:
            raise ValueError(f"ONNX outputs {actual_outputs} do not match policy.yaml {expected_outputs}.")

    @staticmethod
    def _policy_input_to_builder_key(name: str) -> str | None:
        if name in {"obs", "observation.obs"} or name.endswith(".obs"):
            return "obs"
        if name in {"proprio_hist", "observation.proprio_hist"} or name.endswith(".proprio_hist"):
            return "proprio_hist"
        return None
