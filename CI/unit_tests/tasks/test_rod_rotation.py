"""
Unit test for the rod rotation task.
"""
import numpy as np
import pytest

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.object_movement.rod_rotation import RotateRod


class TestRodRotation:
    """
    Test suite for the rod rotations.
    """

    @classmethod
    def setup_class(cls):
        """
        Setup the test class.
        """
        cls.task = RotateRod()

        # Colloid through time
        a = 1 / np.sqrt(2)

        cls.task._historic_rod_director = np.array([a, -a, 0])
        colloid_1 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([1, 0, 0]), type=1
        )
        colloid_2 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([a, a, 0]), type=1
        )
        colloid_3 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([0, 1, 0]), type=1
        )
        colloid_4 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([-a, a, 0]), type=1
        )
        colloid_5 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([-1, 0, 0]), type=1
        )
        colloid_6 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([-a, -a, 0]), type=1
        )
        colloid_7 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([0, -1, 0]), type=1
        )
        colloid_8 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([a, -a, 0]), type=1
        )
        colloid_9 = Colloid(
            pos=np.array([0, 0, 0]), id=0, director=np.array([1, 0, 0]), type=1
        )
        cls.colloids = [
            colloid_1,
            colloid_2,
            colloid_3,
            colloid_4,
            colloid_5,
            colloid_6,
            colloid_7,
            colloid_8,
            colloid_9,
        ]

    def test_rotation_velocity(self):
        """
        Test that the rod rotation velocity is correct.
        """
        reference_velocity = np.deg2rad(45)

        # Test that the velocity is correct
        for colloid in self.colloids:
            colloid_list = [colloid]
            velocity = self.task._compute_angular_velocity(colloid_list[0].director)
            assert velocity == pytest.approx(reference_velocity)
