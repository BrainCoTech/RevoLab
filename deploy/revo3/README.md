# Revo3 deployment tooling

The v1 asset migration invalidates the previous policies, joint calibration and demonstrations. **Hardware policy control is disabled pending new policy validation, axis/zero calibration and a five-fingertip contact provider.** No real robot was connected during migration.

Install the inference tools with `pip install -e deploy/revo3`; SDK access additionally requires the `[hardware]` extra (`bc-stark-sdk==1.4.5`).

The supported export path is HORA Stage 2, from the repository root:

```bash
python scripts/hora/export_onnx.py --stage stage2 \
  --checkpoint /path/to/stage2_nn/model_best.ckpt --output /path/to/policy.onnx
```

Keep `params/asset_contract.json` beside the source run. Export copies that contract into the `.deploy_meta.yaml`. The runtime checks the asset revision and joint order and uses the task's recorded control limits, including HORA limit scaling.

Stage 2 inputs are `obs[B,141]` and `proprio_hist[B,30,47]`: 21 normalized positions, 21 targets and 5 contact values per frame. Actor observations zero contact channels; adaptation history retains them, matching training. The inference API requires explicit contact values. The old RSL exporter fabricated a different Stage 2 contract and has been removed. RSL policies can still be exported for simulation by their RSL play script.

`config/revo3_right.yaml` now contains limits from the new USD. Its zero offsets are **uncalibrated placeholders** (`hardware_calibrated: false`). The CLI supports diagnostic `--dry-run` only: it reads joint state through the SDK, uses explicitly announced zero contact placeholders, and sends no motor commands. This does not validate policy quality. The calibration flag alone does not enable control; a contact provider is also required.

```bash
python deploy/revo3/scripts/run_policy.py \
  --onnx /path/to/policy.onnx --policy /path/to/policy.deploy_meta.yaml \
  --profile deploy/revo3/config/revo3_right.yaml --dry-run
```

Dry-run connects to hardware for reads. Use synthetic inputs and `Stage2InputBuilder` for fully offline checks.
