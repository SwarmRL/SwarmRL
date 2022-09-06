"""
Test the Gumbel distribution.
"""
import numpy as np
from numpy.testing import assert_array_almost_equal

from swarmrl.tasks.searching.gradient_sensing import GradientSensing


class TestGradientSensing:
    """
    Test suite for the run and tumble task.
    """

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
        cls.task = GradientSensing(
            source=np.array([500.0, 500.0, 0.0]),
            decay_function=cls.scale_function,
            reward_scale_factor=10,
            box_size=np.array([1000.0, 1000.0, 1000.0]),
        )

    def test_change_source(self):
        """
        Test if the changing source method actually changes the source.
        """
        _ = self.task.change_source(new_source=np.array([0.0, 0.0, 0.0]))
        assert_array_almost_equal(self.task.source, np.array([0.0, 0.0, 0.0]))

    def test_init_linear_change(self):
        """
        Test how the reward scales with a linear change function and a particle moving
        directly towards the source.
        """
        _ = self.task.init_task()
