"""
Unit test for the position observable.
"""

import numpy as np
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.position import PositionObservable


class TestPositionObservable:
    """
    Test suite for position observable.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        cls.task = PositionObservable(
            box_length=np.array([1.0, 1.0, 1.0]),
            particle_type=0,
        )
        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        cls.colloids = [colloid_1, colloid_2, colloid_3]

    def test_compute_observable(self):
        """
        Test the computation of the observable for all colloids.
        """
        expected = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        actual = self.task.compute_observable(self.colloids)

        assert_array_equal(expected, actual)

    def test_compute_single_observable(self):
        """
        Test the computation of the observable for a single colloid.
        """
        expected = np.array([0.0, 0.0, 0.0])
        actual = self.task.compute_single_observable(0, self.colloids)

        assert_array_equal(expected, actual)
