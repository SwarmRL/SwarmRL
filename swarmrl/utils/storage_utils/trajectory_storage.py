"""Trajectory storages for agent, transition, and simulation data."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from swarmrl.utils.storage_utils.core_storage import HDF5TrajectoryStorage


class ConfigurableTrajectoryStorage(HDF5TrajectoryStorage):
    """Intermediate base class to handle attribute validation and filtering."""

    ALLOWED_FIELDS: set[str] = set()
    PRESETS: dict[str, Sequence[str]] = {}

    def __init__(
        self,
        out_folder: str,
        filename: str,
        h5_group_tag: str,
        preset: str = "minimal",
        stored_attributes: Sequence[str] | None = None,
        allow_existing_file: bool = False,
        write_chunk_size: int = 1,
    ):
        """
        Initialize agent trajectory storage.

        Parameters
        ----------
        out_folder : str (default="./Agent_Data")
            Output folder path.
        filename : str
            filename of the stored Agent file.
        h5_group_tag : str
            Tag for hdf5 storage structure
        preset : str (default="minimal")
            Preset for storage: "minimal" or "all".
            Ignored if stored_attributes is provided.
        stored_attributes : Sequence[str] | None (default=None)
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
                att for att in normalized_attributes if att not in self.ALLOWED_FIELDS
            ]
            if unknown_attributes:
                raise ValueError(
                    f"Unknown stored_attributes: {unknown_attributes}. "
                    f"Allowed: {sorted(self.ALLOWED_FIELDS)}"
                )

            self.stored_attributes = normalized_attributes

        super().__init__(
            out_folder=out_folder,
            filename=filename,
            allow_existing_file=allow_existing_file,
            write_chunk_size=write_chunk_size,
        )
        self._h5_group_tag = h5_group_tag


class AgentTrajectoryStorage(ConfigurableTrajectoryStorage):
    """HDF5 storage for agent trajectory data with configurable fields."""

    ALLOWED_FIELDS = {"actions", "log_probs", "rewards", "features", "killed"}
    PRESETS = {
        "minimal": ("actions", "rewards"),
        "all": ("actions", "log_probs", "rewards", "features", "killed"),
        "verbose": ("actions", "log_probs", "rewards", "features", "killed"),
    }

    def __init__(
        self,
        particle_type: int,
        out_folder: str = "./Agent_Data",
        preset: str = "minimal",
        stored_attributes: Sequence[str] | None = None,
        allow_existing_file: bool = False,
        write_chunk_size: int = 1,
    ):
        self.particle_type = particle_type
        super().__init__(
            out_folder=out_folder,
            filename=f"agent_data_{particle_type}.hdf5",
            h5_group_tag=f"Agent_{particle_type}",
            preset=preset,
            stored_attributes=stored_attributes,
            allow_existing_file=allow_existing_file,
            write_chunk_size=write_chunk_size,
        )

    def _format_field(self, field_name: str, trajectory: Any) -> np.ndarray | None:
        """Helper to safely extract and format specific fields."""
        if not hasattr(trajectory, field_name):
            return None

        val = getattr(trajectory, field_name)
        if val is None:
            return None

        # Special casing for scalar/boolean values and optional features
        if field_name == "killed":
            return np.asarray([val], dtype=np.bool_)
        if field_name == "features":
            arr = np.asarray(val)
            return arr if arr.size > 0 else None

        return np.asarray(val)

    def _get_dataset_specs(self, trajectory: Any) -> dict[str, dict[str, Any]]:
        specs = {}

        for attr in self.stored_attributes:
            arr = self._format_field(attr, trajectory)
            if arr is not None:
                specs[attr] = {
                    "shape": (1, *arr.shape),
                    "maxshape": (None, *arr.shape),
                    "dtype": arr.dtype,
                }
        return specs

    def _extract_sample(self, trajectory: Any) -> dict[str, Any]:
        sample = {}

        for attr in self.stored_attributes:
            arr = self._format_field(attr, trajectory)
            if arr is not None:
                sample[attr] = arr
        return sample


class TransitionTrajectoryStorage(ConfigurableTrajectoryStorage):
    """HDF5 storage for off-policy transition data."""

    ALLOWED_FIELDS = {
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
    }
    PRESETS = {
        "minimal": (
            "action",
            "reward",
        ),
        "all": (
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
        ),
        "verbose": (
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
        ),
    }

    def __init__(
        self,
        particle_type: int,
        out_folder: str = "./Transition_Data",
        preset: str = "minimal",
        stored_attributes: Sequence[str] | None = None,
        allow_existing_file: bool = False,
        write_chunk_size: int = 1,
    ):
        self.particle_type = particle_type
        super().__init__(
            out_folder=out_folder,
            filename=f"sac_transition_data_{particle_type}.hdf5",
            h5_group_tag=f"SAC_{particle_type}",
            preset=preset,
            stored_attributes=stored_attributes,
            allow_existing_file=allow_existing_file,
            write_chunk_size=write_chunk_size,
        )

    def _format_field(self, field_name: str, transition: Any) -> np.ndarray:
        """Helper to safely extract and format specific fields."""
        if not hasattr(transition, field_name):
            return None
        val = getattr(transition, field_name)
        # Scalars need to be wrapped in an array for HDF5 concatenation
        if field_name in {"reward", "terminated", "truncated"}:
            return np.asarray([val])
        return np.asarray(val)

    def _get_dataset_specs(self, transition: Any) -> dict[str, dict[str, Any]]:
        specs = {}
        for attr in self.stored_attributes:
            arr = self._format_field(attr, transition)
            specs[attr] = {
                "shape": (1, *arr.shape),
                "maxshape": (None, *arr.shape),
                "dtype": arr.dtype,
            }
        return specs

    def _extract_sample(self, transition: Any) -> dict[str, Any]:
        return {
            attr: self._format_field(attr, transition)
            for attr in self.stored_attributes
        }


@dataclass
class StorageConfig:
    """Base Configuration for trajectory storage."""

    out_folder: str
    storage_preset: str = "minimal"
    stored_attributes: list[str] | None = None
    allow_existing_file: bool = False
    write_chunk_size: int = 1


@dataclass
class AgentStorageConfig(StorageConfig):
    out_folder: str = "./agent_data"


@dataclass
class TransitionStorageConfig(StorageConfig):
    out_folder: str = "./transition_data"


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

    def _get_dataset_specs(self, timestep_data: dict) -> dict[str, dict[str, Any]]:
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

    def _extract_sample(self, timestep_data: dict) -> dict[str, Any]:
        return {
            "Times": timestep_data["Times"],
            "Ids": timestep_data["Ids"],
            "Types": timestep_data["Types"],
            "Unwrapped_Positions": timestep_data["Unwrapped_Positions"],
            "Velocities": timestep_data["Velocities"],
            "Directors": timestep_data["Directors"],
        }
