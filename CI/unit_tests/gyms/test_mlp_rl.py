"""
Test the mlp rl module.
"""
import numpy as np

from swarmrl.gyms.mlp_rl import MLPRL

import torch


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
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        c = torch.tensor(3.0, requires_grad=True)
        d = torch.tensor(5.0, requires_grad=True)
        e = torch.tensor(6.0, requires_grad=True)
        f = torch.tensor(7.0, requires_grad=True)

        # target_probs = np.array([[1, 2, 3], [5, 6, 7]])
        target_probs = [[a, d], [b, e], [c, f]]
        # target_values = np.array([[1, 2, 3], [5, 6, 7]])
        target_values = [[a, d], [b, e], [c, f]]

        # target_rewards = np.array([[1, 2, 3], [5, 6, 7]])
        target_rewards = [[a, d], [b, e], [c, f]]

        # 2 particles for 3 time steps
        input_data = [
            [a, a, a, a],
            [d, d, d, d],
            [b, b, b, b],
            [e, e, e, e],
            [c, c, c, c],
            [f, f, f, f],
        ]

        probs, values, rewards, _, time = self.rl_trainer._format_episode_data(
            input_data
        )

        np.testing.assert_array_equal(target_probs, probs)
        np.testing.assert_array_equal(target_values, values)
        np.testing.assert_array_equal(target_rewards, rewards)
        assert time == 3
