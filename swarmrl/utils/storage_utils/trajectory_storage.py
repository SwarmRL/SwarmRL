"""Trajectory storages for agent and simulation data."""

from typing import Any, Callable, Dict, List

import numpy as np

from swarmrl.utils.storage_utils.core_storage import HDF5TrajectoryStorage


class DictTrajectoryStorage(HDF5TrajectoryStorage):
    """
    Generic dict-based trajectory storage.

    Provides:
    - a dataset spec builder from one sample
    - a sample extractor that returns a dict keyed like the dataset names
    """

    def __init__(
        self,
        out_folder: str,
        filename: str,
        h5_group_tag: str,
        dataset_specs_builder: Callable[[Any], Dict[str, Dict[str, Any]]],
        sample_extractor: Callable[[Any], Dict[str, Any]],
    ):
        super().__init__(out_folder=out_folder, filename=filename)
        self._h5_group_tag = h5_group_tag
        self._dataset_specs_builder = dataset_specs_builder
        self._sample_extractor = sample_extractor
        self._dataset_keys: List[str] = []

    def _get_dataset_specs(self, data_sample: Any) -> Dict[str, Dict[str, Any]]:
        specs = self._dataset_specs_builder(data_sample)
        self._dataset_keys = list(specs.keys())
        return specs

    def _initialize_data_holder(self) -> Dict[str, List]:
        return {key: list() for key in self._dataset_keys}

    def _accumulate_data(self, data: Any) -> None:
        sample = self._sample_extractor(data)

        if not self._data_holder:
            self._dataset_keys = list(sample.keys())
            self._data_holder = self._initialize_data_holder()

        for key in self._dataset_keys:
            self._data_holder[key].append(sample[key])


class AgentTrajectoryStorage(DictTrajectoryStorage):
    """HDF5 storage for agent trajectory data with configurable fields."""

    ALLOWED_FIELDS = {
        "actions",
        "log_probs",
        "rewards",
        "features",
        "killed",
        "particle_type",
    }

    PRESETS = {
        "minimal": ["actions", "rewards"],
        "verbose": [
            "actions",
            "log_probs",
            "rewards",
            "features",
            "killed",
            "particle_type",
        ],
    }

    def __init__(
        self,
        particle_type: int,
        out_folder: str = "./Agent_Data",
        preset: str = "minimal",
        stored_fields: list = None,
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
            Preset for storage: "minimal" or "verbose".
            Ignored if stored_fields is provided.
        stored_fields : list (default=None)
            Explicit whitelist of fields to store
            (e.g., ["actions", "features"]).
            Overrides preset if provided.
        """
        if stored_fields is None:
            if preset not in self.PRESETS:
                raise ValueError(f"preset must be one of {list(self.PRESETS.keys())}")
            self.stored_fields = list(self.PRESETS[preset])
        else:
            if not isinstance(stored_fields, (list, tuple, set)):
                raise TypeError(
                    "stored_fields must be a list, tuple, or set of field names"
                )

            # Deduplicate while preserving order.
            normalized_fields = list(dict.fromkeys(stored_fields))

            if len(normalized_fields) == 0:
                raise ValueError("stored_fields must contain at least one field")

            unknown_fields = [
                field for field in normalized_fields if field not in self.ALLOWED_FIELDS
            ]
            if unknown_fields:
                raise ValueError(
                    "Unknown stored_fields: "
                    f"{unknown_fields}. Allowed: {sorted(self.ALLOWED_FIELDS)}"
                )

            self.stored_fields = normalized_fields

        super().__init__(
            out_folder=out_folder,
            filename=f"agent_data_{particle_type}.hdf5",
            h5_group_tag=f"Agent_{particle_type}",
            dataset_specs_builder=self._build_agent_specs,
            sample_extractor=self._extract_agent_sample,
        )
        self.particle_type = particle_type

    def _build_agent_specs(self, trajectory) -> Dict[str, Dict[str, Any]]:
        specs = {}

        if "actions" in self.stored_fields:
            actions = np.asarray(trajectory.actions)
            specs["actions"] = {
                "shape": (1, *actions.shape),
                "maxshape": (None, *actions.shape),
                "dtype": actions.dtype,
            }
        if "log_probs" in self.stored_fields:
            log_probs = np.asarray(trajectory.log_probs)
            specs["log_probs"] = {
                "shape": (1, *log_probs.shape),
                "maxshape": (None, *log_probs.shape),
                "dtype": log_probs.dtype,
            }
        if "rewards" in self.stored_fields:
            rewards = np.asarray(trajectory.rewards)
            specs["rewards"] = {
                "shape": (1, *rewards.shape),
                "maxshape": (None, *rewards.shape),
                "dtype": rewards.dtype,
            }

        if "features" in self.stored_fields:
            if getattr(trajectory, "features", None) is not None:
                features = np.asarray(trajectory.features)
                if features.size > 0:
                    specs["features"] = {
                        "shape": (1, *features.shape),
                        "maxshape": (None, *features.shape),
                        "dtype": features.dtype,
                    }
        if "killed" in self.stored_fields:
            killed = np.asarray([trajectory.killed], dtype=np.bool_)
            specs["killed"] = {
                "shape": (1, 1),
                "maxshape": (None, 1),
                "dtype": killed.dtype,
            }

        if "particle_type" in self.stored_fields:
            particle_type = np.asarray([trajectory.particle_type], dtype=np.int64)
            specs["particle_type"] = {
                "shape": (1, 1),
                "maxshape": (None, 1),
                "dtype": particle_type.dtype,
            }

        return specs

    def _extract_agent_sample(self, trajectory) -> Dict[str, Any]:
        sample = {}

        if "actions" in self.stored_fields:
            sample["actions"] = trajectory.actions
        if "log_probs" in self.stored_fields:
            sample["log_probs"] = trajectory.log_probs
        if "rewards" in self.stored_fields:
            sample["rewards"] = trajectory.rewards
        if "killed" in self.stored_fields:
            sample["killed"] = np.asarray([trajectory.killed], dtype=np.bool_)
        if "particle_type" in self.stored_fields:
            sample["particle_type"] = np.asarray(
                [trajectory.particle_type],
                dtype=np.int64,
            )

        if "features" in self.stored_fields:
            if getattr(trajectory, "features", None) is not None:
                features = np.asarray(trajectory.features)
                if features.size > 0:
                    sample["features"] = trajectory.features

        return sample


class SimulationTrajectoryStorage(DictTrajectoryStorage):
    """HDF5 storage for simulation trajectory data."""

    def __init__(
        self,
        out_folder: str = "./trajectories",
        h5_group_tag: str = "colloids",
    ):
        super().__init__(
            out_folder=out_folder,
            filename="trajectory.hdf5",
            h5_group_tag=h5_group_tag,
            dataset_specs_builder=self._build_simulation_specs,
            sample_extractor=self._extract_simulation_sample,
        )

    @staticmethod
    def _build_simulation_specs(timestep_data: Dict) -> Dict[str, Dict[str, Any]]:
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
    def _extract_simulation_sample(timestep_data: Dict) -> Dict[str, Any]:
        return {
            "Times": timestep_data["Times"],
            "Ids": timestep_data["Ids"],
            "Types": timestep_data["Types"],
            "Unwrapped_Positions": timestep_data["Unwrapped_Positions"],
            "Velocities": timestep_data["Velocities"],
            "Directors": timestep_data["Directors"],
        }
