"""Tests for storage writer behavior."""

from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt

from swarmrl.utils.colloid_utils import TrajectoryInformation
from swarmrl.utils.storage_utils import (
    AgentTrajectoryStorage,
    SimulationTrajectoryStorage,
)


def _make_agent_trajectory(
    particle_type: int,
    episode_length: int,
    n_colloids: int,
    base_value: float,
) -> TrajectoryInformation:
    trajectory = TrajectoryInformation(particle_type=particle_type)
    trajectory.features = np.full(
        (episode_length, n_colloids, 1), base_value, dtype=np.float32
    )
    trajectory.actions = np.full(
        (episode_length, n_colloids), int(base_value), dtype=int
    )
    trajectory.log_probs = np.full(
        (episode_length, n_colloids), base_value + 0.1, dtype=float
    )
    trajectory.rewards = np.full(
        (episode_length, n_colloids), base_value + 0.2, dtype=float
    )
    return trajectory


def _make_sim_timestep(n_colloids: int, time_value: float) -> dict:
    return {
        "Times": np.array([time_value], dtype=float)[:, np.newaxis],
        "Ids": np.arange(n_colloids, dtype=int)[:, np.newaxis],
        "Types": np.zeros((n_colloids, 1), dtype=int),
        "Unwrapped_Positions": np.full((n_colloids, 3), time_value, dtype=float),
        "Velocities": np.full((n_colloids, 3), time_value + 1.0, dtype=float),
        "Directors": np.full((n_colloids, 3), time_value + 2.0, dtype=float),
    }


class TestStorageWriters:
    def test_agent_storage_initializes_and_writes_first_sample(self, tmp_path: Path):
        storage = AgentTrajectoryStorage(particle_type=2, out_folder=str(tmp_path))
        trajectory = _make_agent_trajectory(
            2,
            episode_length=4,
            n_colloids=3,
            base_value=1.0,
        )

        storage.write(trajectory)

        file_path = tmp_path / "agent_data_2.hdf5"
        assert file_path.exists()
        assert storage.is_initialized
        assert storage._write_idx == 1

        with h5py.File(file_path.as_posix(), "r") as h5_file:
            group = h5_file["Agent_2"]
            assert group["features"].shape == (1, 4, 3, 1)
            assert group["actions"].shape == (1, 4, 3)
            assert group["log_probs"].shape == (1, 4, 3)
            assert group["rewards"].shape == (1, 4, 3)
            npt.assert_allclose(group["features"][0], trajectory.features)

    def test_agent_storage_appends_without_overwriting(self, tmp_path: Path):
        storage = AgentTrajectoryStorage(particle_type=0, out_folder=str(tmp_path))
        first = _make_agent_trajectory(
            0, episode_length=2, n_colloids=2, base_value=3.0
        )
        second = _make_agent_trajectory(
            0, episode_length=2, n_colloids=2, base_value=7.0
        )

        storage.write(first)
        storage.write(second)

        file_path = tmp_path / "agent_data_0.hdf5"
        assert storage._write_idx == 2

        with h5py.File(file_path.as_posix(), "r") as h5_file:
            dataset = h5_file["Agent_0"]["features"]
            assert dataset.shape == (2, 2, 2, 1)
            npt.assert_allclose(dataset[0], first.features)
            npt.assert_allclose(dataset[1], second.features)

    def test_sim_storage_batch_write_appends(self, tmp_path: Path):
        storage = SimulationTrajectoryStorage(
            out_folder=str(tmp_path),
            h5_group_tag="traj_group",
        )
        first = _make_sim_timestep(n_colloids=3, time_value=1.0)
        storage.write(first)

        batch = {
            "Times": [
                _make_sim_timestep(3, 2.0)["Times"],
                _make_sim_timestep(3, 3.0)["Times"],
            ],
            "Ids": [
                _make_sim_timestep(3, 2.0)["Ids"],
                _make_sim_timestep(3, 3.0)["Ids"],
            ],
            "Types": [
                _make_sim_timestep(3, 2.0)["Types"],
                _make_sim_timestep(3, 3.0)["Types"],
            ],
            "Unwrapped_Positions": [
                _make_sim_timestep(3, 2.0)["Unwrapped_Positions"],
                _make_sim_timestep(3, 3.0)["Unwrapped_Positions"],
            ],
            "Velocities": [
                _make_sim_timestep(3, 2.0)["Velocities"],
                _make_sim_timestep(3, 3.0)["Velocities"],
            ],
            "Directors": [
                _make_sim_timestep(3, 2.0)["Directors"],
                _make_sim_timestep(3, 3.0)["Directors"],
            ],
        }

        storage.write_accumulated_batch(batch)

        file_path = tmp_path / "trajectory.hdf5"
        with h5py.File(file_path.as_posix(), "r") as h5_file:
            times = h5_file["traj_group"]["Times"]
            assert times.shape[0] == 3
            assert np.isclose(times[0, 0, 0], 1.0)
            assert np.isclose(times[1, 0, 0], 2.0)
            assert np.isclose(times[2, 0, 0], 3.0)

    def test_empty_batch_is_noop(self, tmp_path: Path):
        storage = SimulationTrajectoryStorage(
            out_folder=str(tmp_path),
            h5_group_tag="traj_group",
        )
        storage.write(_make_sim_timestep(n_colloids=2, time_value=1.0))
        idx_before = storage._write_idx

        empty_batch = {
            "Times": [],
            "Ids": [],
            "Types": [],
            "Unwrapped_Positions": [],
            "Velocities": [],
            "Directors": [],
        }
        storage.write_accumulated_batch(empty_batch)

        assert storage._write_idx == idx_before
