# Third-Party Notices

RevoLab's own code is covered by the root [MIT license](LICENSE), except where
individual files retain different copyright or SPDX notices. Preserve those
notices and the applicable license texts when redistributing derived files.

| Component | Location / scope | License and source |
| --- | --- | --- |
| Isaac Lab | Derived files in `source/BrainCo_DexHand/` and training scripts with upstream BSD-3-Clause headers | [BSD-3-Clause text](LICENSES/IsaacLab.txt), [upstream](https://github.com/isaac-sim/IsaacLab) |
| HORA | Adapted learning code in `source/BrainCo_DexHand/BrainCo_DexHand/algo/hora/` | [MIT text, copyright 2022 Haozhi Qi](LICENSES/HORA.txt), [upstream](https://github.com/HaozhiQi/hora) |
| brainco-description | Robot descriptions in the Git submodule | Licensing pending in the pinned upstream release; see below |

The Isaac Lab license text was copied unchanged from the tested runtime commit
[`b4c3210`](https://github.com/isaac-sim/IsaacLab/blob/b4c321024792976150ca55fddb26fa34480d974e/LICENSE).
The HORA text was copied from
[`410d958`](https://github.com/HaozhiQi/hora/blob/410d95824dd28b198f6d6910cb4c09330a6303be/LICENSE),
with trailing whitespace removed and all wording retained.
These commits identify the license snapshots, not the original code-port revisions.

## Robot and task assets

The [brainco-description](https://github.com/BrainCoTech/brainco-description)
submodule is pinned to `6db7e2ed1d864ee0a3ab6ff62d43a303787137f6`. It is not
relicensed under RevoLab MIT. This release contains no LICENSE or COPYING file;
its README states that licensing is pending.

Pre-existing task meshes, point clouds and ground textures are listed in the
[task asset inventory](assets/README.md); their original sources are not recorded
per file. Simulation GIFs depict the upstream robot assets.

Isaac Sim and separately installed dependencies retain their own licenses;
RevoLab's MIT license does not cover those distributions.
