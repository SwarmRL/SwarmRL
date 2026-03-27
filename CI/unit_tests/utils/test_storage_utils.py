"""Tests for storage writer behavior."""

from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import pytest

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
    trajectory.killed = bool(base_value > 5.0)
    return trajectory


def _make_agent_trajectory_vector_features(
    particle_type: int,
    episode_length: int,
    n_colloids: int,
    feature_shape: tuple,
    base_value: float,
) -> TrajectoryInformation:
    trajectory = _make_agent_trajectory(
        particle_type=particle_type,
        episode_length=episode_length,
        n_colloids=n_colloids,
        base_value=base_value,
    )
    full_shape = (episode_length, n_colloids, *feature_shape)
    trajectory.features = np.full(full_shape, base_value, dtype=np.float32)
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
    def test_agent_storage_rejects_unknown_stored_field(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown stored_fields"):
            AgentTrajectoryStorage(
                particle_type=1,
                out_folder=str(tmp_path),
                stored_fields=["actions", "foo"],
            )

    def test_agent_storage_rejects_empty_stored_fields(self, tmp_path: Path):
        with pytest.raises(ValueError, match="at least one field"):
            AgentTrajectoryStorage(
                particle_type=1,
                out_folder=str(tmp_path),
                stored_fields=[],
            )

    def test_agent_storage_rejects_non_iterable_stored_fields(self, tmp_path: Path):
        with pytest.raises(TypeError, match="list, tuple, or set"):
            AgentTrajectoryStorage(
                particle_type=1,
                out_folder=str(tmp_path),
                stored_fields="actions",
            )

    def test_agent_storage_default_skips_features(self, tmp_path: Path):
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
            assert group["actions"].shape == (1, 4, 3)
            assert group["rewards"].shape == (1, 4, 3)
            assert "log_probs" not in group
            assert "features" not in group

    def test_agent_storage_appends_without_overwriting(self, tmp_path: Path):
        storage = AgentTrajectoryStorage(
            particle_type=0,
            out_folder=str(tmp_path),
            preset="verbose",
        )
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

    def test_agent_storage_supports_vector_features_in_verbose_mode(
        self,
        tmp_path: Path,
    ):
        storage = AgentTrajectoryStorage(
            particle_type=3,
            out_folder=str(tmp_path),
            preset="verbose",
        )
        trajectory = _make_agent_trajectory_vector_features(
            particle_type=3,
            episode_length=5,
            n_colloids=4,
            feature_shape=(6,),
            base_value=2.0,
        )

        storage.write(trajectory)

        file_path = tmp_path / "agent_data_3.hdf5"
        with h5py.File(file_path.as_posix(), "r") as h5_file:
            feature_dataset = h5_file["Agent_3"]["features"]
            assert feature_dataset.shape == (1, 5, 4, 6)
            npt.assert_allclose(feature_dataset[0], trajectory.features)

    def test_agent_storage_custom_fields_whitelist(self, tmp_path: Path):
        storage = AgentTrajectoryStorage(
            particle_type=4,
            out_folder=str(tmp_path),
            stored_fields=["actions", "rewards"],
        )
        trajectory = _make_agent_trajectory(
            4,
            episode_length=3,
            n_colloids=2,
            base_value=5.0,
        )

        storage.write(trajectory)

        file_path = tmp_path / "agent_data_4.hdf5"
        with h5py.File(file_path.as_posix(), "r") as h5_file:
            group = h5_file["Agent_4"]
            assert "actions" in group
            assert "rewards" in group
            assert "log_probs" not in group
            assert "features" not in group

    def test_agent_storage_persists_killed_and_particle_type(self, tmp_path: Path):
        storage = AgentTrajectoryStorage(
            particle_type=4,
            out_folder=str(tmp_path),
            stored_fields=["killed", "particle_type"],
        )
        trajectory = _make_agent_trajectory(
            4,
            episode_length=3,
            n_colloids=2,
            base_value=6.0,
        )

        storage.write(trajectory)

        file_path = tmp_path / "agent_data_4.hdf5"
        with h5py.File(file_path.as_posix(), "r") as h5_file:
            group = h5_file["Agent_4"]
            assert group["killed"].shape == (1, 1)
            assert group["particle_type"].shape == (1, 1)
            assert bool(group["killed"][0, 0]) is True
            assert int(group["particle_type"][0, 0]) == 4

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
