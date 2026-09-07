"""Trajectory storages for agent and simulation data."""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from swarmrl.utils.storage_utils.core_storage import HDF5TrajectoryStorage


class AgentTrajectoryStorage(HDF5TrajectoryStorage):
    """HDF5 storage for agent trajectory data with configurable fields."""

    ALLOWED_FIELDS = {
        "actions",
        "log_probs",
        "rewards",
        "features",
        "killed",
    }
    PRESETS = {
        "minimal": ("actions", "rewards"),
        "all": (
            "actions",
            "log_probs",
            "rewards",
            "features",
            "killed",
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
            (e.g., ["actions", "features"]).
            Overrides preset if provided.
        allow_existing_file : bool (default=False)
            If False, raise FileExistsError when the target file already exists.
            If True, allow writing to an existing HDF5 file.
        write_chunk_size : int (default=1)
            Number of complete agent trajectory samples buffered before appending to
            HDF5. The default 1 preserves immediate writes.
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
        specs = {}

        if "actions" in self.stored_attributes:
            actions = np.asarray(trajectory.actions)
            specs["actions"] = {
                "shape": (1, *actions.shape),
                "maxshape": (None, *actions.shape),
                "dtype": actions.dtype,
            }
        if "log_probs" in self.stored_attributes:
            log_probs = np.asarray(trajectory.log_probs)
            specs["log_probs"] = {
                "shape": (1, *log_probs.shape),
                "maxshape": (None, *log_probs.shape),
                "dtype": log_probs.dtype,
            }
        if "rewards" in self.stored_attributes:
            rewards = np.asarray(trajectory.rewards)
            specs["rewards"] = {
                "shape": (1, *rewards.shape),
                "maxshape": (None, *rewards.shape),
                "dtype": rewards.dtype,
            }

        if "features" in self.stored_attributes:
            if getattr(trajectory, "features", None) is not None:
                features = np.asarray(trajectory.features)
                if features.size > 0:
                    specs["features"] = {
                        "shape": (1, *features.shape),
                        "maxshape": (None, *features.shape),
                        "dtype": features.dtype,
                    }
        if "killed" in self.stored_attributes:
            killed = np.asarray([trajectory.killed], dtype=np.bool_)
            specs["killed"] = {
                "shape": (1, 1),
                "maxshape": (None, 1),
                "dtype": killed.dtype,
            }

        return specs

    def _extract_sample(self, trajectory) -> Dict[str, Any]:
        sample = {}

        if "actions" in self.stored_attributes:
            sample["actions"] = trajectory.actions
        if "log_probs" in self.stored_attributes:
            sample["log_probs"] = trajectory.log_probs
        if "rewards" in self.stored_attributes:
            sample["rewards"] = trajectory.rewards
        if "killed" in self.stored_attributes:
            sample["killed"] = np.asarray([trajectory.killed], dtype=np.bool_)

        if "features" in self.stored_attributes:
            if getattr(trajectory, "features", None) is not None:
                features = np.asarray(trajectory.features)
                if features.size > 0:
                    sample["features"] = trajectory.features

        return sample


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
        write_chunk_size: int = 1,
    ):
        super().__init__(
            out_folder=out_folder,
            filename="trajectory.hdf5",
            allow_existing_file=allow_existing_file,
            write_chunk_size=write_chunk_size,
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
