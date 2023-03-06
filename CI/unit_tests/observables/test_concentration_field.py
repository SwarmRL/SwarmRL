"""
Unit test for the concentration field observable.
"""
import numpy as np
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.concentration_field import ConcentrationField


class TestConcentrationField:
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

        cls.observable = ConcentrationField(
            source=np.array([0.5, 0.5, 0.0]),
            decay_fn=decay_fn,
            box_length=np.array([1.0, 1.0, 1.0]),
            particle_type=0,
        )
        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        cls.colloids = [colloid_1, colloid_2, colloid_3]

    def test_init(self):
        """
        Test if the observable is initialized correctly.
        """
        self.observable.initialize(colloids=self.colloids)

        # Test the concentration field initialization.
        assert_array_equal(self.observable.source, np.array([0.5, 0.5, 0.0]))
        assert_array_equal(self.observable.box_length, np.array([1.0, 1.0, 1.0]))
        assert self.observable.decay_fn(1) == -1
        assert self.observable.scale_factor == 100.0
        assert_array_equal(
            list(self.observable._historic_positions.keys()), ["0", "1", "2"]
        )
        assert_array_equal(
            self.observable._historic_positions["0"], np.array([0.0, 0.0, 0.0])
        )
        assert_array_equal(
            self.observable._historic_positions["1"], np.array([0.0, 1.0, 0.0])
        )
        assert_array_equal(
            self.observable._historic_positions["2"], np.array([1.0, 1.0, 0.0])
        )

    def test_compute_observable(self):
        """
        Test if the observable is computed correctly.
        """
        self.observable.initialize(colloids=self.colloids)

        colloid_1 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        new_colloids = [colloid_1, colloid_2, colloid_3]

        observables = self.observable.compute_observable(colloids=new_colloids)

        distance_colloid_1 = np.linalg.norm(colloid_1.pos - self.observable.source)
        distance_colloid_2 = np.linalg.norm(colloid_2.pos - self.observable.source)
        distance_colloid_3 = np.linalg.norm(colloid_3.pos - self.observable.source)

        delta_colloid_1 = distance_colloid_1 - np.linalg.norm(
            self.colloids[0].pos - self.observable.source
        )
        delta_colloid_2 = distance_colloid_2 - np.linalg.norm(
            self.colloids[1].pos - self.observable.source
        )
        delta_colloid_3 = distance_colloid_3 - np.linalg.norm(
            self.colloids[2].pos - self.observable.source
        )

        observables_should_be = np.array(
            [
                -1 * self.observable.scale_factor * delta_colloid_1,
                -1 * self.observable.scale_factor * delta_colloid_2,
                -1 * self.observable.scale_factor * delta_colloid_3,
            ]
        ).reshape(-1, 1)
        assert_array_equal(observables, observables_should_be)
