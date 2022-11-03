"""
Unit test for the vision cone.
"""

import numpy as np

from swarmrl.observables.vision_cone import VisionCone


class TestVisionCone:
    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        cls.vc = VisionCone(
            vision_angle=90,
            vision_range=1,
            return_cone=False,
            vision_direction=complex(0, 1),
        )

    def test_call(self):
        """
        Tests if the computation of the observable is still correct.
        """

        class Colloid:
            def __init__(self, position, director):
                self.pos = position
                self.director = director

        col = Colloid(np.array([0.5, 0.5, 0]), np.array([0.0, 1.0, 0]))
        col2 = Colloid(np.array([0.5, 0.8, 0]), np.array([0.0, 1.0, 0]))
        col3 = Colloid(np.array([0.5, 0.4, 0]), np.array([0.0, 1.0, 0]))
        other_colloids = [col2, col3]

        assert self.vc.compute_observable(col, other_colloids)[0] == 0.3
