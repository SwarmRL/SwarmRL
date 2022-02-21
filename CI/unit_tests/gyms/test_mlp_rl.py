"""
Test the mlp rl module.
"""
import numpy as np

from swarmrl.gyms.mlp_rl import MLPRL


class TestMLPRL:
    """
    Test the MLP RL module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test class.
        """
        cls.rl_trainer = MLPRL(
            actor=None,
            critic=None,
            task=None,
            loss=None,
            observable=None,
            n_particles=2,
        )

    def test_format_episode_data(self):
        """
        Test that the format data method is working correctly.
        """
        # target_probs = np.array([[1, 2, 3], [5, 6, 7]])
        target_probs = np.array([[1, 5], [2, 6], [3, 7]])
        # target_values = np.array([[1, 2, 3], [5, 6, 7]])
        target_values = np.array([[1, 5], [2, 6], [3, 7]])

        # target_rewards = np.array([[1, 2, 3], [5, 6, 7]])
        target_rewards = np.array([[1, 5], [2, 6], [3, 7]])

        # 2 particles for 3 time steps
        input_data = [
                [1, 1, 1],
                [5, 5, 5],
                [2, 2, 2],
                [6, 6, 6],
                [3, 3, 3],
                [7, 7, 7],
            ]

        probs, values, rewards, _ = self.rl_trainer._format_episode_data(input_data)

        np.testing.assert_array_equal(target_probs, probs)
        np.testing.assert_array_equal(target_values, values)
        np.testing.assert_array_equal(target_rewards, rewards)
