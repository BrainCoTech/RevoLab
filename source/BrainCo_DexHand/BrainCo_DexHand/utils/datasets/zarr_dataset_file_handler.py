# Copyright (c) 2024-2026, The BrainCo Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local Zarr dataset handler for diffusion-policy style exports.

This implementation is adapted from UWLab's Zarr dataset handler so BrainCo can
export `.zarr` demonstration datasets without depending on a local UWLab checkout.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import torch
import zarr
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData


class ZarrDatasetFileHandler(DatasetFileHandlerBase):
    """Store episode data in Zarr format compatible with diffusion_policy."""

    def __init__(self, chunk_size: int = 5000, image_chunk_size: int = 50, image_keys: list[str] | None = None):
        self._dataset = None
        self._dataset_path = None
        self._env_name = None
        self._episode_count = 0
        self._chunk_size = chunk_size
        self._image_chunk_size = image_chunk_size
        self._image_keys = image_keys
        self._compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    def create(self, file_path: str, env_name: str | None = None, overwrite: bool = True):
        """Create a new `.zarr` dataset on disk."""
        if not file_path.endswith(".zarr"):
            raise ValueError("Dataset file path must end with .zarr")

        self._dataset_path = Path(file_path)
        if self._dataset_path.exists():
            if not overwrite:
                raise ValueError(f"Dataset already exists at {self._dataset_path}")
            print(f"Removing existing dataset at {self._dataset_path}", flush=True)
            shutil.rmtree(self._dataset_path)

        self._env_name = env_name or "isaac_lab_env"
        self._task_description = "Custom task"

        try:
            self._dataset = zarr.group(str(self._dataset_path))
            self._dataset.create_group("data")
            meta_group = self._dataset.create_group("meta")
            meta_group.zeros("episode_ends", shape=(0,), dtype=np.int64, compressor=None)
            self._dataset.attrs["env_name"] = self._env_name
            self._dataset.attrs["task_description"] = self._task_description
        except Exception as exc:
            raise RuntimeError(f"Failed to create Zarr dataset: {exc}") from exc

        self._episode_count = 0

    def open(self, file_path: str, mode: str = "r"):
        raise NotImplementedError("Open not implemented for Zarr handler")

    def get_env_name(self) -> str | None:
        return self._env_name

    def get_episode_names(self) -> Iterable[str]:
        if self._dataset is None:
            return []
        return [f"episode_{i:06d}" for i in range(self._episode_count)]

    def get_num_episodes(self) -> int:
        return self._episode_count

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        if self._dataset is None or episode.is_empty():
            return

        self._convert_and_save_episode(episode)
        if demo_id is None:
            self._episode_count += 1

    def _convert_and_save_episode(self, episode: EpisodeData):
        episode_dict = episode.data
        if "actions" not in episode_dict or "obs" not in episode_dict:
            raise ValueError("Episode must contain actions and observations")

        num_frames = episode_dict["actions"].shape[0]
        processed_obs = self._process_observations_for_episode(episode_dict["obs"])

        episode_data = {
            "actions": episode_dict["actions"].cpu().numpy(),
            "obs": processed_obs,
            "rewards": episode_dict.get("rewards", torch.zeros(num_frames)).cpu().numpy(),
            "dones": episode_dict.get("dones", torch.cat([torch.zeros(num_frames - 1), torch.ones(1)])).cpu().numpy(),
        }
        self._save_episode_to_zarr(episode_data)

    def _process_observations_for_episode(self, obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        episode_obs = {}
        for obs_key, value in obs_dict.items():
            try:
                episode_obs[obs_key] = value.cpu().numpy()
            except Exception as exc:
                print(f"Error processing observation '{obs_key}': {exc}", flush=True)
        return episode_obs

    def _save_episode_to_zarr(self, episode_data: dict[str, Any]):
        if self._dataset is None:
            raise RuntimeError("Dataset not initialized")

        data_group = self._dataset["data"]
        meta_group = self._dataset["meta"]
        episode_ends = meta_group["episode_ends"]

        current_end = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
        episode_length = len(episode_data["actions"])
        new_end = current_end + episode_length

        for key, value in episode_data.items():
            if key == "obs":
                for obs_key, obs_value in value.items():
                    self._extend_or_create_array(data_group, f"obs/{obs_key}", obs_value, episode_length)
            else:
                self._extend_or_create_array(data_group, key, value, episode_length)

        episode_ends.resize(len(episode_ends) + 1)
        episode_ends[-1] = int(new_end)

    def _extend_or_create_array(self, group, key: str, data: np.ndarray, episode_length: int):
        if key in group:
            arr = group[key]
            arr.resize(arr.shape[0] + episode_length, *arr.shape[1:])
            arr[-episode_length:] = data
        else:
            if self._is_image_array(data):
                chunks = (self._image_chunk_size,) + data.shape[1:]
            else:
                chunks = (self._chunk_size,) + data.shape[1:]
            group.create_dataset(key, data=data, chunks=chunks, dtype=data.dtype, compressor=self._compressor)

    def _is_image_array(self, data: np.ndarray) -> bool:
        if self._image_keys is not None:
            return False
        return data.ndim == 4 and data.shape[-1] in [1, 3, 4]

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        raise NotImplementedError("Load episode not implemented for Zarr handler")

    def flush(self):
        pass

    def close(self):
        self._dataset = None

    def add_env_args(self, env_args: dict):
        pass
