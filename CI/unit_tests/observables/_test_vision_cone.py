"""
Unit test for the vision cone.
"""

from dataclasses import dataclass

import numpy as np

from swarmrl.observables._vision_cone import VisionCone


@dataclass
class Colloid:
    pos: np.ndarray
    director: np.ndarray


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
            source=np.array([0.0, 0.0, 0.0]),
            detect_source=False,
            box_size=np.array([1.0, 1.0, 0.0]),
        )

    def test_call(self):
        """
        Tests if the computation of the observable is still correct.
        """
        col = Colloid(np.array([0.5, 0.5, 0]), np.array([0.0, 1.0, 0]))
        col2 = Colloid(np.array([0.5, 0.8, 0]), np.array([0.0, 1.0, 0]))
        col3 = Colloid(np.array([0.5, 0.4, 0]), np.array([0.0, 1.0, 0]))
        other_colloids = [col2, col3]

        assert self.vc.compute_observable(col, other_colloids) == 0.3
