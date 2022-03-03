"""
Run a unit test on the loss module.
"""
import unittest

import torch

from swarmrl.loss_models.loss import Loss


class TestLoss(unittest.TestCase):
    """
    Test the loss functions for RL models.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test class.

        Returns
        -------

        """
        cls.loss = Loss(n_colloids=10)
        cls.rewards = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
                [8, 8, 8, 8, 8],
                [9, 9, 9, 9, 9],
                [10, 10, 10, 10, 10],
            ]
        )

    def test_expected_returns(self):
        """
        Test the true values of the actual returns.

        Notes
        -------
        Checks the actual values return in the discounted returns without
        standardization and with the simple case of gamma=1
        """
        self.loss.particles = 10
        true_values = torch.tensor(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [10.0, 8.0, 6.0, 4.0, 2.0],
                [15.0, 12.0, 9.0, 6.0, 3.0],
                [20.0, 16.0, 12.0, 8.0, 4.0],
                [25.0, 20.0, 15.0, 10.0, 5.0],
                [30.0, 24.0, 18.0, 12.0, 6.0],
                [35.0, 28.0, 21.0, 14.0, 7.0],
                [40.0, 32.0, 24.0, 16.0, 8.0],
                [45.0, 36.0, 27.0, 18.0, 9.0],
                [50.0, 40.0, 30.0, 20.0, 10.0],
            ]
        )
        discounted_returns = self.loss.compute_discounted_returns(
            rewards=self.rewards, standardize=False, gamma=1
        )
        torch.testing.assert_allclose(true_values, discounted_returns)

    def test_expected_returns_standardized(self):
        """
        Test the expected return method for standardized data.

        Notes
        -----
        Test that the expected returns are correct.
        """
        discounted_returns = self.loss.compute_discounted_returns(
            rewards=self.rewards, standardize=True
        )
        self.assertAlmostEqual(torch.mean(discounted_returns[0]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[1]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[2]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[3]).numpy(), 0.0)
        self.assertAlmostEqual(torch.mean(discounted_returns[4]).numpy(), 0.0)
        self.assertAlmostEqual(torch.std(discounted_returns[0]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[1]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[2]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[3]).numpy(), 1.0)
        self.assertAlmostEqual(torch.std(discounted_returns[4]).numpy(), 1.0)

    def test_actor_loss(self):
        """
        Test the actor loss for 2 particles.

        Returns
        -------

        """
        self.loss.particles = 2
        rewards = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])

        action_probs = torch.nn.Softmax()(
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
        )

        predicted_rewards = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        actor_loss = self.loss.compute_actor_loss(
            policy_probabilities=action_probs,
            rewards=rewards,
            predicted_rewards=predicted_rewards,
        )

        self.assertEqual(129.41423116696092, float(actor_loss[0].numpy()))
        self.assertEqual(len(actor_loss), 2)
        self.assertEqual(actor_loss[0], actor_loss[1])

    def test_critic_loss(self):
        """
        Test the critic loss for 2 particles.

        Returns
        -------

        """
        self.loss.particles = 2
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
        predicted_rewards = torch.tensor(
            [[2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0]]
        )
        critic_loss = self.loss.compute_critic_loss(
            rewards=rewards, predicted_rewards=predicted_rewards
        )
        self.assertEqual(len(critic_loss), 2)
        self.assertEqual(critic_loss[0], critic_loss[1])
        self.assertEqual(critic_loss[0], 6.702364444732666)
