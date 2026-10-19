# Task assets

This directory contains task objects and scenery. Robot descriptions belong in
the pinned `third_party/brainco-description` submodule; see the
[asset contract](../docs/assets.md).

| Directory | Contents |
| --- | --- |
| `urdf/objects/` | Cube URDFs and associated object meshes and materials |
| `urdf/objects/ball/` | Pre-existing ball point cloud files and preview image |
| `usd/ground/` | Ground USD and its texture files |

HORA's ball and cylinder are created procedurally in the task code. Their color
blocks add visual meshes while retaining the original analytic colliders.
Generated grasp caches belong in ignored artifact directories, not here.
