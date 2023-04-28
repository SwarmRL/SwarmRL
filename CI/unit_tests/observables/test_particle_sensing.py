"""
Unit test for the particle sensing observable.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.particle_sensing import ParticleSensing


class TestParticleSensing:
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

        cls.observable = ParticleSensing(
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
        self.observable.initialize(colloids=self.colloids)

        # Test the concentration field initialization.
        assert_array_equal(self.observable.box_length, np.array([1.0, 1.0, 1.0]))
        assert self.observable.decay_fn(1) == -1
        assert self.observable.scale_factor == 100.0
        assert self.observable.sensing_type == 0

        assert_array_equal(
            list(self.observable.historical_field.keys()), ["0", "1", "2"]
        )

        straight_distance = -2.0
        triangular_distance = -np.sqrt(2) - 1.0

        assert self.observable.historical_field["0"] == straight_distance
        assert self.observable.historical_field["1"] == pytest.approx(
            triangular_distance
        )
        assert self.observable.historical_field["2"] == pytest.approx(
            triangular_distance
        )

    def test_closer_approach(self):
        """
        Test what happens if a particle gets closer.
        """
        # Initialize with class colloids
        self.observable.scale_factor = 1.0
        self.observable.initialize(colloids=self.colloids)

        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 0.5, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        colloids = [colloid_1, colloid_2, colloid_3]

        observable = self.observable.compute_observable(colloids=colloids)

        assert observable[0] == 0.5
