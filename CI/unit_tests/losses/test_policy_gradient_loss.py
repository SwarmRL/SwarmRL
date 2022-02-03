"""
Run a unit test on the loss module.
"""
import pytest
from swarmrl.losses.policy_gradient_loss import (
    PolicyGradientLoss,
    compute_actor_loss,
    compute_critic_loss,
    compute_true_value_function
)
import torch


class TestLoss:
    """
    Test the loss functions for RL models.
    """

    @classmethod
    def setup_class(cls) -> None:
        """
        Set up the test class.

        Returns
        -------

        """
        cls.loss = PolicyGradientLoss()
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
        target_values = torch.tensor(
            [
                [5., 4., 3., 2., 1.],
                [10., 8., 6., 4., 2.],
                [15., 12., 9., 6., 3.],
                [20., 16., 12., 8., 4.],
                [25., 20., 15., 10., 5.],
                [30., 24., 18., 12., 6.],
                [35., 28., 21., 14., 7.],
                [40., 32., 24., 16., 8.],
                [45., 36., 27., 18., 9.],
                [50., 40., 30., 20., 10.]
            ]
        )
        value_function = compute_true_value_function(
            rewards=self.rewards, gamma=1, standardize=False
        )
        torch.testing.assert_allclose(target_values, value_function)

    def test_expected_returns_standardized(self):
        """
        Test the expected return method for standardized data.

        Notes
        -----
        Test that the expected returns are correct.
        """
        discounted_returns = compute_true_value_function(
            rewards=self.rewards
        )
        torch.testing.assert_allclose(torch.mean(discounted_returns[0]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[1]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[2]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[3]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[4]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[0]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[1]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[2]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[3]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[4]).numpy(), 1.0)

    def test_actor_loss(self):
        """
        Test the actor loss for 2 particles.

        Returns
        -------

        """
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
        actor_loss = compute_actor_loss(
            log_probs=action_probs,
            rewards=rewards,
            predicted_values=predicted_rewards
        )

        assert float(actor_loss[0].numpy()) == pytest.approx(5.455, 0.001)
        assert len(actor_loss) == 2
        assert actor_loss[0] == actor_loss[1]

    def test_critic_loss(self):
        """
        Test the critic loss for 2 particles.

        Returns
        -------

        """
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
        critic_loss = compute_critic_loss(
            rewards=rewards,
            predicted_rewards=predicted_rewards
        )
        assert len(critic_loss) == 2
        assert critic_loss[0] == critic_loss[1]
        assert critic_loss[0] == pytest.approx(3.5, 0.001)
