"""
Unit test for the rod rotation task.
"""

import numpy as np
import pytest

from swarmrl.components import Colloid
from swarmrl.tasks.object_movement.rod_rotation import RotateRod


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


class TestRodRotation:
    """
    Test suite for the rod rotations.
    """

    @classmethod
    def setup_class(cls):
        """
        Setup the test class.
        """
        cls.reference_velocty = 45.0

    def test_ccw_rotation(self):
        """
        Setup the test class.
        """
        task = RotateRod(direction="CCW", angular_velocity_scale=1.0)

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(self.reference_velocty)

        # Test negative rewards.
        task = RotateRod(direction="CCW", angular_velocity_scale=1.0)
        starting_director, colloids = create_trajectory(direction_scale=-1)

        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(-1 * self.reference_velocty)

    def test_cw_rotation(self):
        """
        Setup the test class.
        """
        task = RotateRod(direction="CW", angular_velocity_scale=1.0)

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=-1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(self.reference_velocty)

        # Test negative rewards.
        task = RotateRod(direction="CW", angular_velocity_scale=1.0)
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)
            assert velocity == pytest.approx(-1 * self.reference_velocty)
