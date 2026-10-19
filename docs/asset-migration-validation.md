# Asset migration validation — 2026-10-16

Branch: `codex/asset-refresh-retrain`, based on RevoLab main `bc2f874a61f386712374888675e6756316d0aadc`.
Robot descriptions: `6db7e2ed1d864ee0a3ab6ff62d43a303787137f6`.

## Verified locally

- Submodule initialization and Git LFS retrieval using `bash scripts/setup_assets.sh`; submodule worktree remains clean.
- Offline USD reference, joint/body count and fixed-frame checks with `usd-core 26.8`. `OmniPBR.mdl` is supplied by Isaac Sim and is the only allowed standalone-USD resolver exception.
- Editable package installation as `BrainCo_DexHand 0.2.0`.
- Python compilation and two offline deployment-input regression tests.
- Six GPU environments, each with **2 environments × 32 zero-action steps**, including reset, finite observations/rewards, physical and control limits, palm frames, and artifact metadata round-trip/mismatch rejection:

| Task | Runtime joints / actions | Result |
| --- | ---: | --- |
| Repose Cube v1 | 21 / 21 | Passed |
| Reorient Cylinder v1 | 21 / 21 | Passed |
| HORA Ball v1 | 21 / 21 | Passed, default-pose reset |
| HORA Cylinder v1 | 21 / 21 | Passed, default-pose reset |
| Right Lift v1 | 28 / 28 | Passed |
| Dynamic Handover Cube v1 | 56 / 56 | Passed, all cloned torso/head joints verified as fixed |

- HORA Stage 2 export and ONNX Runtime inference with an **untrained random test model**. Exported metadata preserved the asset contract and control limits. Inputs were 141-dimensional observations and 30 × 47 history; output was 21 finite joint targets. No SDK was connected.

Local runtime: Ubuntu 22.04, RTX 5090, Isaac Lab checkout VERSION `2.3.2`, Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`, Python 3.11, PyTorch `2.7.0+cu128`, RSL-RL `5.0.1`, RL-Games `1.6.1`, Gymnasium `1.2.1`, ONNX Runtime `1.27.0`. The Isaac Sim build is a release candidate; these results do not establish support for other simulator builds.

Raw local logs are ignored under `tmp/asset-migration/`: `offline-final.log`, `final-<task-id>.log`, `handover5.log`, and `export-test.log`. The committed validation commands are in README.md; the synthetic weights are disposable test artifacts, not policies for use.

## Issues resolved during validation

- Fixed-frame tensor slices required `reshape`, replacing assumptions of contiguous rigid-body state arrays.
- Applying a joint change after spawning must cover every cloned environment. Zero-width limits still allowed numerical drift; native fixed joints remove the three unused DOFs entirely.
- Old deployment code used 126/30×42 inputs while HORA training uses 141/30×47. Deployment input construction now retains five contact channels in adaptation history and zeros them only for the Stage 2 actor.
- The old RSL deployment exporter fabricated a HORA Stage 2 contract. It was removed; the actual HORA exporter now includes the source run's physical contract.

## Deferred work

No optimizer steps, training runs, policy-performance evaluation, successful-grasp certification, or physical hardware commands were performed. New HORA caches, six retrained policies (including HORA Stage 2), deployment calibration/contact input integration, and new demonstrations remain for subsequent work. The upstream team also needs to clarify the robot asset license. The GitHub workflow was added; its checks were run locally, not dispatched remotely.
