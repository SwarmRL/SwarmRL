"""
Test the Gumbel distribution.
"""
from swarmrl.tasks.searching.gradient_sensing import GradientSensing


class TestGradientSensing:
    """
    Test suite for the run and tumble task.
    """

    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """

        cls.task = GradientSensing(reward_scale_factor=1)

    def test_linear_change(self):
        """
        Test how the reward scales with a linear change function and a particle moving
        directly towards the source.
        """

        def scale_fn(distance: float):
            """
            A linear scale function w.r.t distance.
            """
            return 1 - distance

        self.task.decay_fn = scale_fn
        # observable = self.task.init_task()
