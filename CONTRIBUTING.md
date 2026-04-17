# Contributing to BrainCo Isaac Lab

Thank you for your interest in contributing to BrainCo's public Isaac Lab task suite.
This repository is maintained by BrainCo, Inc. and is released under the MIT License.

## Ways to Contribute

- Report bugs and unexpected behavior
- Propose improvements to existing tasks, assets, or documentation
- Submit pull requests with fixes or enhancements

## Task Naming Convention

Public Revo3 task IDs follow the `BrainCo-<framework>-<robot>-<task>-v0`
convention (for example, `BrainCo-Direct-Revo3-Repose-Cube-v0` or `BrainCo-Dexsuite-Revo3-Right-Lift-v0`). New task
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
Instead, report them privately to the maintainers. We will
acknowledge receipt within a reasonable timeframe and work with you on a fix
and coordinated disclosure.

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

## Maintainers

This project is maintained by the BrainCo Robotics team.

- General contact: opensource@brainco.tech
- Security contact: security@brainco.tech
