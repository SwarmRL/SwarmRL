"""
Tests for the MultiSensing class.
"""

import numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.concentration_field import ConcentrationField
from swarmrl.observables.director import Director
from swarmrl.observables.multi_sensing import MultiSensing
from swarmrl.observables.position import PositionObservable


class TestMultiSensing:
    """
    Test suite for the MultiSensing class.
    """

    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        # Define colloids in the box.
        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        cls.colloids = [colloid_1, colloid_2, colloid_3]

        # Define the concentration field observable
        def decay_fn(x: float):
            """
            Scaling function for the test

            Parameters
            ----------
            x : float
                    Input value.
            """
            return -1 * x

        cls.concentration_observable = ConcentrationField(
            source=np.array([0.5, 0.5, 0.0]),
            decay_fn=decay_fn,
            box_length=np.array([1.0, 1.0, 1.0]),
            particle_type=0,
        )

        # Define position observable.
        cls.position_observable = PositionObservable(
            box_length=np.array([1.0, 1.0, 1.0])
        )

        # Define the director observable.
        cls.director_observable = Director()

        # Define the multisensing observable for all other observables
        cls.observable = MultiSensing(
            observables=[
                cls.concentration_observable,
                cls.position_observable,
                cls.director_observable,
            ],
        )

    def test_compute_observable(self):
        """
        Tests the computation of all observables.
        """
        # Initialize the observables
        self.observable.initialize(self.colloids)

        # Define new colloid positions
        colloid_1 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        new_colloids = [colloid_1, colloid_2, colloid_3]

        # True value for concentration field
        distance_colloid_1 = np.linalg.norm(
            colloid_1.pos - self.concentration_observable.source
        )
        distance_colloid_2 = np.linalg.norm(
            colloid_2.pos - self.concentration_observable.source
        )
        distance_colloid_3 = np.linalg.norm(
            colloid_3.pos - self.concentration_observable.source
        )

        delta_colloid_1 = distance_colloid_1 - np.linalg.norm(
            self.colloids[0].pos - self.concentration_observable.source
        )
        delta_colloid_2 = distance_colloid_2 - np.linalg.norm(
            self.colloids[1].pos - self.concentration_observable.source
        )
        delta_colloid_3 = distance_colloid_3 - np.linalg.norm(
            self.colloids[2].pos - self.concentration_observable.source
        )

        concentration_should_be = [
            -1 * self.concentration_observable.scale_factor * delta_colloid_1,
            -1 * self.concentration_observable.scale_factor * delta_colloid_2,
            -1 * self.concentration_observable.scale_factor * delta_colloid_3,
        ]

        # True value for position observable
        position_should_be = [colloid_1.pos, colloid_2.pos, colloid_3.pos]

        # True value for the director observable
        director_should_be = [
            colloid_1.director,
            colloid_2.director,
            colloid_3.director,
        ]

        # Compute the observables
        observable = self.observable.compute_observable(new_colloids)

        # Check shape of output
        shape_should_be = (3, 3)
        assert np.shape(observable) == shape_should_be
        assert np.shape(observable[0][0]) == (1,)
        assert np.shape(observable[0][1]) == (3,)
        assert np.shape(observable[0][2]) == (3,)

        # Check values of the outputs.

        # Concentration field
        assert np.allclose(observable[0][0], concentration_should_be[0])
        assert np.allclose(observable[1][0], concentration_should_be[1])
        assert np.allclose(observable[2][0], concentration_should_be[2])

        # Position
        assert np.allclose(observable[0][1], position_should_be[0])
        assert np.allclose(observable[1][1], position_should_be[1])
        assert np.allclose(observable[2][1], position_should_be[2])

        # Director
        assert np.allclose(observable[0][2], director_should_be[0])
        assert np.allclose(observable[1][2], director_should_be[1])
        assert np.allclose(observable[2][2], director_should_be[2])
