"""
Test the random exploration module.
"""
import pytest

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

    def test_definite_action(self):
        """
        Force specific actions and ensure it works.
        """
        # Force a new point
        self.explorer.probability = 1.0
        chosen_action = self.explorer(model_action=2.3, action_space_length=4)
        assert chosen_action != 2.3

        # Force to keep the model action
        self.explorer.probability = 0.0
        chosen_action = self.explorer(model_action=2.3, action_space_length=4)
        assert chosen_action == 2.3

    def test_distribution(self):
        """
        Test that the correct exploration distribution is produced.
        """
        self.explorer.probability = 0.6

        exploration_steps = 0
        for i in range(1000):
            chosen_action = self.explorer(model_action=2.3, action_space_length=4)
            if chosen_action != 2.3:
                exploration_steps += 1

        assert exploration_steps / 1000 == pytest.approx(0.6, 0.1)
