"""
Unit test for the rod torque task.
"""

import numpy as np
import pytest

from swarmrl.components import Colloid
from swarmrl.tasks.object_movement.rod_torque import RodTorque


def create_trajectory(direction_scale: int = 1):
    """
    Create a trajectory for the tests.
    """
    colloids = []
    angle = 0.0

    starting_director = np.array([1, 0, 0])
    for _ in range(100):
        angle += np.deg2rad(direction_scale * 45)
        director = np.array([np.cos(angle), np.sin(angle), 0])
        colloids.append(
            Colloid(pos=np.array([0, 0, 0]), id=0, director=director, type=1)
        )

    return starting_director, colloids


class TestRodTorque:
    """
    Test suite for the rod torque task.
    """

    @classmethod
    def setup_class(cls):
        """
        Setup the test class.
        """
        cls.reference_velocity = 45.0

    def test_ccw_rotation(self):
        """
        Setup the test class.
        """
        task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=1
        )

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(self.reference_velocity)

        # Test opposite rotation direction.
        task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=1
        )
        starting_director, colloids = create_trajectory(direction_scale=-1)

        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(0.0)

    def test_cw_rotation(self):
        """
        Setup the test class.
        """
        task = RodTorque(
            direction="CW", angular_velocity_scale=1.0, velocity_history_size=1
        )

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=-1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(self.reference_velocity * -1)

        # Test opposite rotation direction.
        task = RodTorque(
            direction="CW", angular_velocity_scale=1.0, velocity_history_size=1
        )
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)
            assert velocity == pytest.approx(0.0)

    def test_velocity_history(self):
        """
        Setup the test class.
        """
        task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=100
        )
        # The history buffer is initialized with nans and filled from the end,
        # so the reference is built the same way and compared with nanmean.
        velocity_history = np.full(100, np.nan)

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)
            velocity_history = np.roll(velocity_history, -1)
            velocity_history[-1] = self.reference_velocity
            assert velocity == pytest.approx(np.nanmean(velocity_history))

    def test_compute_torque_on_rod(self):
        """
        Setup the test class.
        """
        task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=1
        )

        torque = task._compute_torque_on_rod(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[0.0, 1.0, 0.0]]),
            colloid_positions=np.array([[1.0, -1.0, 0.0]]),
        )
        assert torque == pytest.approx([-1])
        torque = task._compute_torque_on_rod(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[0.0, -1.0, 0.0]]),
            colloid_positions=np.array([[-1.0, 1.0, 0.0]]),
        )
        assert torque == pytest.approx([-1])
        torque = task._compute_torque_on_rod(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[1.0, 0.0, 0.0]]),
            colloid_positions=np.array([[100.0, 100.0, 0.0]]),
        )
        assert torque == pytest.approx([0])
        torque = task._compute_torque_on_rod(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[0.0, 1.0, 0.0]]),
            colloid_positions=np.array([[-1.0, -1.0, 0.0]]),
        )
        assert torque == pytest.approx([1])
