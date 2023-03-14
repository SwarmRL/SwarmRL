"""
Test the random exploration module.
"""
import jax.numpy as np
from numpy.testing import assert_array_equal, assert_raises

from swarmrl.exploration_policies.random_exploration import RandomExploration


class TestRandomExploration:
    """
    Test suite for the random exploration module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for testing.
        """
        cls.explorer = RandomExploration(probability=0.8)

        cls.chosen_actions = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def test_definite_action(self):
        """
        Force specific actions and ensure the exploration is performed.
        """
        # Force a new point
        self.explorer.probability = 1.0
        chosen_actions = self.explorer(
            model_actions=self.chosen_actions, action_space_length=4
        )
        assert_raises(
            AssertionError, assert_array_equal, chosen_actions, self.chosen_actions
        )

    def test_no_change(self):
        """
        Test that the chosen actions are not changed when the probability is 0.
        """
        self.explorer.probability = 0.0

        for i in range(10):
            chosen_actions = self.explorer(
                model_actions=self.chosen_actions, action_space_length=4
            )
            assert_array_equal(chosen_actions, self.chosen_actions)
