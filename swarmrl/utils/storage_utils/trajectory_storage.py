"""Trajectory storages for agent and simulation data."""

from typing import Any, Callable, Dict, List

import numpy as np

from swarmrl.utils.storage_utils.core_storage import HDF5TrajectoryStorage


class DictTrajectoryStorage(HDF5TrajectoryStorage):
    """
    Generic dict-based trajectory storage.

    Concrete users provide:
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
    """HDF5 storage for agent trajectory data."""

    def __init__(
        self,
        particle_type: int,
        out_folder: str = "./Agent_Data",
    ):
        super().__init__(
            out_folder=out_folder,
            filename=f"agent_data_{particle_type}.hdf5",
            h5_group_tag=f"Agent_{particle_type}",
            dataset_specs_builder=self._build_agent_specs,
            sample_extractor=self._extract_agent_sample,
        )
        self.particle_type = particle_type

    @staticmethod
    def _build_agent_specs(trajectory) -> Dict[str, Dict[str, Any]]:
        n_colloids = np.array(trajectory.features).shape[1]
        episode_length = np.array(trajectory.features).shape[0]

        return {
            "features": {
                "shape": (1, episode_length, n_colloids, 1),
                "maxshape": (None, episode_length, n_colloids, 1),
                "dtype": np.float32,
            },
            "actions": {
                "shape": (1, episode_length, n_colloids),
                "maxshape": (None, episode_length, n_colloids),
                "dtype": int,
            },
            "log_probs": {
                "shape": (1, episode_length, n_colloids),
                "maxshape": (None, episode_length, n_colloids),
                "dtype": float,
            },
            "rewards": {
                "shape": (1, episode_length, n_colloids),
                "maxshape": (None, episode_length, n_colloids),
                "dtype": float,
            },
        }

    @staticmethod
    def _extract_agent_sample(trajectory) -> Dict[str, Any]:
        return {
            "features": trajectory.features,
            "actions": trajectory.actions,
            "log_probs": trajectory.log_probs,
            "rewards": trajectory.rewards,
        }


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
