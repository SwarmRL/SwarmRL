"""
Test the Gumbel distribution.
"""
import numpy as np
from numpy.testing import assert_array_equal

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.searching.gradient_sensing import GradientSensing


class TestGradientSensing:
    """
    Test suite for the run and tumble task.
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

        cls.task = GradientSensing(
            source=np.array([0.5, 0.5, 0.0]),
            decay_function=decay_fn,
            box_length=np.array([1.0, 1.0, 1.0]),
            particle_type=0,
        )
        colloid_1 = Colloid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        cls.colloids = [colloid_1, colloid_2, colloid_3]

    def test_init(self):
        """
        Test if the task is initialized correctly.
        """
        self.task.initialize(colloids=self.colloids)

        # Test the concentration field initialization.
        assert_array_equal(self.task.source, np.array([0.5, 0.5, 0.0]))
        assert_array_equal(self.task.box_length, np.array([1.0, 1.0, 1.0]))
        assert self.task.decay_fn(1) == -1
        assert self.task.reward_scale_factor == 10.0
        assert_array_equal(list(self.task._historic_positions.keys()), ["0", "1", "2"])
        assert_array_equal(
            self.task._historic_positions["0"], np.array([0.0, 0.0, 0.0])
        )
        assert_array_equal(
            self.task._historic_positions["1"], np.array([0.0, 1.0, 0.0])
        )
        assert_array_equal(
            self.task._historic_positions["2"], np.array([1.0, 1.0, 0.0])
        )

    def test_call_method(self):
        """
        Test if the reward is computed correctly.
        """
        self.task.initialize(colloids=self.colloids)

        colloid_1 = Colloid(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0]), 0, 0)
        colloid_2 = Colloid(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 1, 0)
        colloid_3 = Colloid(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0]), 2, 0)

        new_colloids = [colloid_1, colloid_2, colloid_3]

        rewards = self.task(colloids=new_colloids)

        distance_colloid_1 = np.linalg.norm(colloid_1.pos - self.task.source)
        distance_colloid_2 = np.linalg.norm(colloid_2.pos - self.task.source)
        distance_colloid_3 = np.linalg.norm(colloid_3.pos - self.task.source)

        delta_colloid_1 = distance_colloid_1 - np.linalg.norm(
            self.colloids[0].pos - self.task.source
        )
        delta_colloid_2 = distance_colloid_2 - np.linalg.norm(
            self.colloids[1].pos - self.task.source
        )
        delta_colloid_3 = distance_colloid_3 - np.linalg.norm(
            self.colloids[2].pos - self.task.source
        )

        rewads_should_be = [
            -1 * self.task.reward_scale_factor * delta_colloid_1,
            -1 * self.task.reward_scale_factor * delta_colloid_2,
            -1 * self.task.reward_scale_factor * delta_colloid_3,
        ]

        assert rewards == rewads_should_be
