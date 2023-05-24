"""
Test suite for the species search.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.searching.species_search import SpeciesSearch


class TestSpeciesSearch:
    """
    Test suite for the gradient sensing observable.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """

        def decay_fn(x: float):
            """
            Scaling function for the test

            Parameters
            ----------
            x : float
                    Input value.
            """
            return -1 * x

        cls.task = SpeciesSearch(
            decay_fn=decay_fn,
            box_length=np.array([1.0, 1.0, 1.0]),
            particle_type=0,
        )
        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        cls.colloids = [colloid_1, colloid_2, colloid_3]

    def test_init(self):
        """
        Test if the observable is initialized correctly.
        """
        self.task.initialize(colloids=self.colloids)

        # Test the concentration field initialization.
        assert_array_equal(self.task.box_length, np.array([1.0, 1.0, 1.0]))
        assert self.task.decay_fn(1) == -1
        assert self.task.scale_factor == 100.0
        assert self.task.sensing_type == 0

        assert_array_equal(list(self.task.historical_field.keys()), ["0", "1", "2"])

        straight_distance = -2.0
        triangular_distance = -np.sqrt(2) - 1.0

        assert self.task.historical_field["0"] == straight_distance
        assert self.task.historical_field["1"] == pytest.approx(triangular_distance)
        assert self.task.historical_field["2"] == pytest.approx(triangular_distance)

        # Test that a second initialization leaves evrything fixed.
        self.task.initialize(colloids=self.colloids)
        assert self.task.historical_field["0"] == straight_distance
        assert self.task.historical_field["1"] == pytest.approx(triangular_distance)
        assert self.task.historical_field["2"] == pytest.approx(triangular_distance)

    def test_closer_approach(self):
        """
        Test what happens if a particle gets closer.
        """
        # Initialize with class colloids
        self.task.scale_factor = 1.0
        self.task.initialize(colloids=self.colloids)

        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 0.5, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        colloids = [colloid_1, colloid_2, colloid_3]

        reward = self.task(colloids=colloids)

        assert reward[0] == 0.5

    def test_movement_away(self):
        """
        Test what happens if a particle gets closer.
        """
        # Initialize with class colloids
        self.task.scale_factor = 1.0
        self.task.initialize(colloids=self.colloids)

        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 1.5, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        colloids = [colloid_1, colloid_2, colloid_3]

        reward = self.task(colloids=colloids)

        assert reward[0] == 0.0

    def test_double_call(self):
        """
        Test what happens if you call the task many times.
        """
        # Initialize with class colloids
        self.task.scale_factor = 1.0
        self.task.initialize(colloids=self.colloids)

        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 0.5, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        colloids = [colloid_1, colloid_2, colloid_3]

        reward = self.task(colloids=colloids)

        assert reward[0] == 0.5
        for _ in range(5):
            reward = self.task(colloids=colloids)
            assert reward[0] == 0.0
