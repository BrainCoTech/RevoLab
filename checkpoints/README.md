# Policy artifacts

All six v1 tasks have successful simulation demonstrations from the 2026-10-19
training campaign; see the [README](../README.md#simulation-demos). Training has
stopped and checkpoints are retained in the experiment environments, but no v1
pretrained weights are published in this repository or as a release yet.
HORA demonstrations use Stage 1 teachers; Stage 2 has not been trained.

Generated weights belong in ignored `logs/` or `outputs/` directories. Preserve
each run's `params/asset_contract.json` and training configuration with its
checkpoints. A future weight release should identify the source Git commit,
robot submodule commit, task ID, training stage, evaluation conditions and
artifact license. Old v0 weights and grasp caches are incompatible with v1.
