# Documentation

The root [README](../README.md) is the current task and simulation-demo summary.
All six tasks have successful demos from the 2026-10-19 training campaign;
training has stopped. HORA Stage 2 and hardware validation remain outside that
campaign's scope, and pretrained weights have not been published.

## Setup and interfaces

- [Asset contracts and submodule updates](assets.md)
- [Pinned remote runtime and launch commands](remote-training.md)
- [Checkpoint publication status](../checkpoints/README.md)
- [Hardware deployment tooling and its current limits](../deploy/revo3/README.md)
- [Contributing and validation commands](../CONTRIBUTING.md)
- [Licenses and third-party notices](../THIRD_PARTY_NOTICES.md)
- [Security reporting](../SECURITY.md)

## Dated validation and experiment records

These are snapshots of what was tested at the stated date. Their original
deferred-work lists are historical, not the current training backlog. Raw logs,
checkpoints and exploratory scripts under `tmp/` or `outputs/` are local
artifacts and are not included in a fresh clone.

- [Initial asset migration checks — 2026-10-16](asset-migration-validation.md)
- [Three-server runtime checks](remote-validation.md)
- [Task-scene corrections — 2026-10-17](task-scene-validation.md)
- [Palm collision experiment — 2026-10-19](palm-collision-experiment.md)
- [Collision equivalence audit — 2026-10-19](collision-equivalence-audit.md)

The collision experiments were not adopted; production tasks still use the
pinned robot collision geometry. Successful policy GIFs are in `media/`.
