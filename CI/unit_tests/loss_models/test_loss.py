"""
Run a unit test on the loss module.
"""
import unittest
from swarmrl.loss_models.loss import Loss
import torch


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
                [10, 10, 10, 10, 10]
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
        true_values = torch.tensor([[5., 4., 3., 2., 1.],
                                    [10., 8., 6., 4., 2.],
                                    [15., 12., 9., 6., 3.],
                                    [20., 16., 12., 8., 4.],
                                    [25., 20., 15., 10., 5.],
                                    [30., 24., 18., 12., 6.],
                                    [35., 28., 21., 14., 7.],
                                    [40., 32., 24., 16., 8.],
                                    [45., 36., 27., 18., 9.],
                                    [50., 40., 30., 20., 10.]])
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
        rewards = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]
            ]
        )

        action_probs = torch.nn.Softmax()(
            torch.tensor(
                [
                    [1., 2., 3., 4., 5.],
                    [1., 2., 3., 4., 5.]
                ]
            )
        )

        predicted_rewards = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]
            ]
        )
        actor_loss = self.loss.compute_actor_loss(
            policy_probabilities=action_probs,
            rewards=rewards,
            predicted_rewards=predicted_rewards
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
        rewards = torch.tensor(
            [
                [1., 2., 3., 4., 5.],
                [1., 2., 3., 4., 5.]
            ]
        )
        predicted_rewards = torch.tensor(
            [
                [2., 3., 4., 5., 6.],
                [2., 3., 4., 5., 6.]
            ]
        )
        critic_loss = self.loss.compute_critic_loss(
            rewards=rewards,
            predicted_rewards=predicted_rewards
        )
        self.assertEqual(len(critic_loss), 2)
        self.assertEqual(critic_loss[0], critic_loss[1])
        self.assertEqual(critic_loss[0], 6.702364444732666)

    def test_reward_trajectory(self):
        """
        Test that high rewards result in lower losses.

        Returns
        -------

        """
        self.loss.particles = 2
        rewards = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1]
            ]
        )

        action_probs = torch.nn.Softmax()(
            torch.tensor(
                [
                    [0.1, .2, .3, .1, 0.],
                    [.1, .2, .3, .1, 0.]
                ]
            )
        )

        predicted_rewards = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]
        )

        actor_loss = self.loss.actor_loss(
            policy_probabilities=action_probs,
            rewards=rewards,
            predicted_rewards=predicted_rewards
        )

        print(actor_loss)
