"""
Unit test for the subdivided vision cones.
"""

import numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.subdivided_vision_cones import SubdividedVisionCones


class TestSubdividedVisionCones:
    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        cls.vc = SubdividedVisionCones(
            vision_range=10,
            vision_half_angle=np.pi / 2,
            n_cones=3,
            radii=[1, 2, 3, 4, 1],
            particle_type=0,
        )

    def test_call(self):
        """
        Tests if the computation of the observable is still correct.
        """

        col0 = Colloid(np.array([0, 0, 0]), np.array([0, 1.0, 0]), 0, 0)
        col1 = Colloid(np.array([0, 5, 0]), np.array([1.0, 0, 0]), 1, 0)
        col2 = Colloid(np.array([0, 8, 0]), np.array([1.0, 0, 0]), 2, 1)
        col3 = Colloid(np.array([-7, 8, 0]), np.array([0.0, 1.0, 0]), 3, 1)
        col4 = Colloid(np.array([1, 1, 0]), np.array([0.0, 1.0, 0]), 4, 0)
        colloids = [col0, col1, col2, col3, col4]
        observables = self.vc.compute_observable(colloids)
        observable = observables[0]
        assert observable[0, 0] == 1.0
        assert observable[1, 0] == 0.8
        assert observable[2, 0] == 0.0
        assert observable[0, 1] == 0.0
        assert observable[1, 1] == 0.75
        assert observable[2, 1] == 0.0
