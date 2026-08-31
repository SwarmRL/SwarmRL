"""Trajectory storages for agent and simulation data."""

from dataclasses import dataclass
from typing import Any, Dict

import h5py
import numpy as np

from swarmrl.utils.storage_utils.core_storage import HDF5TrajectoryStorage


class AgentTrajectoryStorage(HDF5TrajectoryStorage):
    """HDF5 storage for agent trajectory data with configurable fields."""

    TIME_ALIGNED_FIELDS = {
        "actions",
        "log_probs",
        "rewards",
        "features",
        "terminated",
        "truncated",
    }

    ALLOWED_FIELDS = {
        "actions",
        "log_probs",
        "rewards",
        "features",
        "terminated",
        "truncated",
        "final_observation",
    }
    PRESETS = {
        "minimal": ("actions", "rewards"),
        "all": (
            "actions",
            "log_probs",
            "rewards",
            "features",
            "terminated",
            "truncated",
            "final_observation",
        ),
    }
    PRESETS["verbose"] = PRESETS["all"]

    def __init__(
        self,
        particle_type: int,
        out_folder: str = "./Agent_Data",
        preset: str = "minimal",
        stored_attributes: list = None,
        allow_existing_file: bool = False,
        write_chunk_size: int = 1,
    ):
        """
        Initialize agent trajectory storage.

        Parameters
        ----------
        particle_type : int
            Particle type ID.
        out_folder : str (default="./Agent_Data")
            Output folder path.
        preset : str (default="minimal")
            Preset for storage: "minimal" or "all".
            Ignored if stored_attributes is provided.
        stored_attributes : list (default=None)
            Explicit whitelist of attributes to store
            (e.g., ["actions", "features", "terminated", "truncated",
            "final_observation"]).
            Overrides preset if provided.
        allow_existing_file : bool (default=False)
            If False, raise FileExistsError when the target file already exists.
            If True, allow writing to an existing HDF5 file.
        write_chunk_size : int (default=1)
            Number of complete agent trajectory samples buffered before appending to
            HDF5. The default 1 preserves immediate writes.

        Notes
        -----
        Time-aligned trajectory fields may have different rollout (trainer episode)
        lengths and are padded in storage. The ``trajectory_length`` dataset records
        each original length. ``final_observation`` stores the bootstrap-only
        observation after the last transition.
        """
        if stored_attributes is None:
            if preset not in self.PRESETS:
                raise ValueError(f"preset must be one of {list(self.PRESETS.keys())}")
            self.stored_attributes = list(self.PRESETS[preset])
        else:
            if not isinstance(stored_attributes, (list, tuple, set)):
                raise TypeError(
                    "stored_attributes must be a list, tuple, or set of "
                    "attribute names"
                )

            # Deduplicate while preserving order.
            normalized_attributes = list(dict.fromkeys(stored_attributes))

            if len(normalized_attributes) == 0:
                raise ValueError(
                    "stored_attributes must contain at least one attribute"
                )

            unknown_attributes = [
                attribute
                for attribute in normalized_attributes
                if attribute not in self.ALLOWED_FIELDS
            ]
            if unknown_attributes:
                raise ValueError(
                    "Unknown stored_attributes: "
                    f"{unknown_attributes}. "
                    f"Allowed: {sorted(self.ALLOWED_FIELDS)}"
                )

            self.stored_attributes = normalized_attributes

        super().__init__(
            out_folder=out_folder,
            filename=f"agent_data_{particle_type}.hdf5",
            allow_existing_file=allow_existing_file,
            write_chunk_size=write_chunk_size,
        )
        self._h5_group_tag = f"Agent_{particle_type}"
        self.particle_type = particle_type

    def _get_dataset_specs(self, trajectory) -> Dict[str, Dict[str, Any]]:
        trajectory_length = np.asarray(len(trajectory.rewards), dtype=np.int64)
        specs = {
            "trajectory_length": {
                "shape": (1,),
                "maxshape": (None,),
                "dtype": trajectory_length.dtype,
            }
        }

        if "actions" in self.stored_attributes:
            actions = np.asarray(trajectory.actions)
            specs["actions"] = {
                "shape": (1, *actions.shape),
                "maxshape": (None, None, *actions.shape[1:]),
                "dtype": actions.dtype,
            }
        if "log_probs" in self.stored_attributes:
            log_probs = np.asarray(trajectory.log_probs)
            specs["log_probs"] = {
                "shape": (1, *log_probs.shape),
                "maxshape": (None, None, *log_probs.shape[1:]),
                "dtype": log_probs.dtype,
            }
        if "rewards" in self.stored_attributes:
            rewards = np.asarray(trajectory.rewards)
            specs["rewards"] = {
                "shape": (1, *rewards.shape),
                "maxshape": (None, None, *rewards.shape[1:]),
                "dtype": rewards.dtype,
            }

        if "features" in self.stored_attributes:
            if getattr(trajectory, "features", None) is not None:
                features = np.asarray(trajectory.features)
                if features.size > 0:
                    specs["features"] = {
                        "shape": (1, *features.shape),
                        "maxshape": (None, None, *features.shape[1:]),
                        "dtype": features.dtype,
                    }
        if "terminated" in self.stored_attributes:
            terminated = np.asarray(trajectory.terminated, dtype=np.bool_)
            specs["terminated"] = {
                "shape": (1, *terminated.shape),
                "maxshape": (None, None, *terminated.shape[1:]),
                "dtype": terminated.dtype,
            }
        if "truncated" in self.stored_attributes:
            truncated = np.asarray(trajectory.truncated, dtype=np.bool_)
            specs["truncated"] = {
                "shape": (1, *truncated.shape),
                "maxshape": (None, None, *truncated.shape[1:]),
                "dtype": truncated.dtype,
            }
        if "final_observation" in self.stored_attributes:
            final_observation = np.asarray(trajectory.final_observation)
            specs["final_observation"] = {
                "shape": (1, *final_observation.shape),
                "maxshape": (None, *final_observation.shape),
                "dtype": final_observation.dtype,
            }

        return specs

    def _extract_sample(self, trajectory) -> Dict[str, Any]:
        sample = {
            "trajectory_length": np.asarray(len(trajectory.rewards), dtype=np.int64)
        }

        if "actions" in self.stored_attributes:
            sample["actions"] = trajectory.actions
        if "log_probs" in self.stored_attributes:
            sample["log_probs"] = trajectory.log_probs
        if "rewards" in self.stored_attributes:
            sample["rewards"] = trajectory.rewards
        if "terminated" in self.stored_attributes:
            sample["terminated"] = np.asarray(trajectory.terminated, dtype=np.bool_)
        if "truncated" in self.stored_attributes:
            sample["truncated"] = np.asarray(trajectory.truncated, dtype=np.bool_)
        if "final_observation" in self.stored_attributes:
            sample["final_observation"] = trajectory.final_observation

        if "features" in self.stored_attributes:
            if getattr(trajectory, "features", None) is not None:
                features = np.asarray(trajectory.features)
                if features.size > 0:
                    sample["features"] = trajectory.features

        return sample

    def _write_to_h5(self) -> None:
        """Write agent trajectories, padding only their variable time axis."""
        if not self._data_holder or all(
            len(values) == 0 for values in self._data_holder.values()
        ):
            return

        n_new = len(next(iter(self._data_holder.values())))
        with h5py.File(self.h5_filename.as_posix(), "a", libver="latest") as h5_file:
            group = h5_file[self._h5_group_tag]

            for key, buffered_values in self._data_holder.items():
                dataset = group[key]
                dataset.resize(self._write_idx + n_new, axis=0)

                if key in self.TIME_ALIGNED_FIELDS:
                    arrays = [np.asarray(value) for value in buffered_values]
                    max_length = max(
                        dataset.shape[1], *(array.shape[0] for array in arrays)
                    )
                    if max_length > dataset.shape[1]:
                        dataset.resize(max_length, axis=1)

                    for offset, array in enumerate(arrays):
                        row = self._write_idx + offset
                        dataset[row, : array.shape[0]] = array
                else:
                    dataset[self._write_idx : self._write_idx + n_new] = np.stack(
                        buffered_values, axis=0
                    )

            self._write_idx += n_new

        self._data_holder = self._initialize_data_holder()


@dataclass
class AgentStorageConfig:
    """Configuration for optional agent trajectory storage."""

    out_folder: str = "./agent_data"
    storage_preset: str = "minimal"
    stored_attributes: list[str] | None = None
    allow_existing_file: bool = False
    write_chunk_size: int = 1


class SimulationTrajectoryStorage(HDF5TrajectoryStorage):
    """HDF5 storage for simulation trajectory data."""

    def __init__(
        self,
        out_folder: str = "./trajectories",
        h5_group_tag: str = "colloids",
        allow_existing_file: bool = False,
    ):
        super().__init__(
            out_folder=out_folder,
            filename="trajectory.hdf5",
            allow_existing_file=allow_existing_file,
        )
        self._h5_group_tag = h5_group_tag

    @staticmethod
    def _get_dataset_specs(timestep_data: Dict) -> Dict[str, Dict[str, Any]]:
        n_particles = len(timestep_data.get("Ids", []))

        return {
            "Times": {
                "shape": (1, 1, 1),
                "maxshape": (None, 1, 1),
                "dtype": float,
            },
            "Ids": {
                "shape": (1, n_particles, 1),
                "maxshape": (None, n_particles, 1),
                "dtype": int,
            },
            "Types": {
                "shape": (1, n_particles, 1),
                "maxshape": (None, n_particles, 1),
                "dtype": int,
            },
            "Unwrapped_Positions": {
                "shape": (1, n_particles, 3),
                "maxshape": (None, n_particles, 3),
                "dtype": float,
            },
            "Velocities": {
                "shape": (1, n_particles, 3),
                "maxshape": (None, n_particles, 3),
                "dtype": float,
            },
            "Directors": {
                "shape": (1, n_particles, 3),
                "maxshape": (None, n_particles, 3),
                "dtype": float,
            },
        }

    @staticmethod
    def _extract_sample(timestep_data: Dict) -> Dict[str, Any]:
        return {
            "Times": timestep_data["Times"],
            "Ids": timestep_data["Ids"],
            "Types": timestep_data["Types"],
            "Unwrapped_Positions": timestep_data["Unwrapped_Positions"],
            "Velocities": timestep_data["Velocities"],
            "Directors": timestep_data["Directors"],
        }
