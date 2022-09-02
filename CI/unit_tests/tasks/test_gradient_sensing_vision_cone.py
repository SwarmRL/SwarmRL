import numpy as np
from numpy.testing import assert_array_almost_equal

from swarmrl.tasks.searching import GradientSensingVisionCone


class TestGSVC:
    def scale_function(distance: float):
        """
        Scaling function for the task
        """
        return 1 - distance

    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        cls.task = GradientSensingVisionCone(
            source=np.array([500.0, 500.0, 0.0]),
            decay_function=cls.scale_function,
            grad_reward_scale_factor=10,
            box_size=np.array([1000.0, 1000.0, 1000.0]),
            cone_reward_scale_factor=0.01,
            vision_angle=60,
            vision_direction=complex(0, 1),
        )

    def test_change_source(self):
        """
        Test if the changing source method actually changes the source.
        """
        _ = self.task.change_source(new_source=np.array([0.0, 0.0, 0.0]))
        assert_array_almost_equal(self.task.source, np.array([0.0, 0.0, 0.0]))
