# Contributing to RevoLab

Thank you for your interest in contributing to BrainCo's public Isaac Lab task suite.
This repository is maintained by BrainCo, Inc. Its own code uses the MIT License;
retain the exceptions documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Ways to Contribute

- Report bugs and unexpected behavior
- Propose improvements to existing tasks, assets, or documentation
- Submit pull requests with fixes or enhancements

## Task Naming Convention

Public Revo3 task IDs follow the `BrainCo-<framework>-<robot>-<task>-v1`
convention (for example, `BrainCo-Direct-Revo3-Repose-Cube-v1` or `BrainCo-Dexsuite-Revo3-Right-Lift-v1`). New task
registrations contributed to this repository should follow the same pattern
so that IDs remain consistent across frameworks and robots.

## Reporting Issues

Please open a GitHub issue with:

- A clear title and description
- Steps to reproduce, including the exact task ID and command
- Your Isaac Lab / Isaac Sim version, OS, Python version, and GPU
- Relevant logs or stack traces


## Reporting Security Vulnerabilities

Please do **not** file public GitHub issues for security problems.
Follow the private contact instructions in [SECURITY.md](SECURITY.md).

## Pull Requests

1. Fork the repository and create a topic branch from `main`.
2. Keep changes focused and include a clear description of intent.
3. Preserve behavior of published task IDs and checkpoints unless the change
   explicitly documents a breaking update.
4. Ensure your changes do not introduce absolute paths, personal data, or
   machine-specific configuration into tracked files.
5. By submitting a pull request, you agree that your contribution will be
   licensed under the MIT License used by this repository.

## Code Style

- Follow the style of existing files in `source/BrainCo_DexHand/`.
- Preserve upstream Isaac Lab copyright and SPDX headers where present.
- Do not modify third-party source headers without a corresponding update to
  `THIRD_PARTY_NOTICES.md`.

## Support
Need help? Reach out through the following channels:
- Submit a ticket: [https://web.static.brainco.cn/work-order](https://web.static.brainco.cn/work-order)
- GitHub: [https://github.com/BrainCoTech](https://github.com/BrainCoTech)

## Validation

In a separate Python 3.11 environment with `usd-core==26.8`, NumPy and PyYAML,
run the same offline checks as CI after initializing the asset submodule:

```bash
bash scripts/setup_assets.sh
python scripts/check_assets.py
python -m compileall -q source scripts deploy
python -m unittest discover -s tests -v
git diff --check
```

Use the Isaac Lab runtime for GPU checks described in README.md. Do not install
standalone `usd-core` into that simulation environment. Report the commands and
results actually run; an offline pass does not establish training performance.

## Asset changes

Robot descriptions belong in brainco-description. Update its pinned submodule commit and the task adapters together; do not vendor robot meshes into RevoLab. Run the offline asset check and GPU smoke checks described in README.md. A physics change requires reviewing all dependent policy contracts.
