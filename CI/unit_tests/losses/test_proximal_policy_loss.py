"""
Run a unit test on the loss module.
"""
import pytest
import torch

from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss


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
        cls.loss.n_particles = 10
        cls.loss.n_time_steps = 5
        cls.rewards = torch.transpose(
            torch.tensor(
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
            ),
            0,
            1,
        )

    def test_expected_returns(self):
        """
        Test the true values of the actual returns.

        Notes
        -------
        Checks the actual values return in the discounted returns without
        standardization and with the simple case of gamma=1
        """
        target_values = torch.transpose(
            torch.tensor(
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
            ),
            0,
            1,
        )
        value_function = self.loss.compute_true_value_function(
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
        discounted_returns = self.loss.compute_true_value_function(rewards=self.rewards)
        torch.testing.assert_allclose(torch.mean(discounted_returns[:, 0]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[:, 1]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[:, 2]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[:, 3]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.mean(discounted_returns[:, 4]).numpy(), 0.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[:, 0]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[:, 1]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[:, 2]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[:, 3]).numpy(), 1.0)
        torch.testing.assert_allclose(torch.std(discounted_returns[:, 4]).numpy(), 1.0)

    def test_actor_loss(self):
        """
        Test the actor loss for 2 particles.

        Returns
        -------

        """
        # Update class parameters for new case.
        self.loss.n_particles = 2
        self.loss.n_time_steps = 5

        rewards = torch.transpose(
            torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), 0, 1
        )

        action_probs = torch.nn.Softmax()(
            torch.transpose(
                torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]),
                0,
                1,
            )
        )

        predicted_rewards = torch.transpose(
            torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]), 0, 1
        )
        actor_loss = self.loss.compute_actor_loss(
            log_probs=action_probs, rewards=rewards, predicted_values=predicted_rewards
        )

        assert float(actor_loss[0].numpy()) == pytest.approx(7.5, 0.001)
        assert len(actor_loss) == 2
        assert actor_loss[0] == actor_loss[1]

        # Reset to defaults for non-linear deployment case.
        self.loss.n_particles = 10
        self.loss.n_time_steps = 5

    def test_critic_loss(self):
        """
        Test the critic loss for 2 particles.
        """
        # Update class parameters for new case.
        self.loss.n_particles = 2
        self.loss.n_time_steps = 5

        rewards = torch.transpose(
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]), 0, 1
        )

        predicted_rewards = torch.transpose(
            torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0]]), 0, 1
        )
        critic_loss = self.loss.compute_critic_loss(
            rewards=rewards, predicted_rewards=predicted_rewards
        )
        assert len(critic_loss) == 2
        assert critic_loss[0] == critic_loss[1]
        assert critic_loss[0] == pytest.approx(17.5, 0.001)

        # Reset to defaults for asymmetric deployment case.
        self.loss.n_particles = 10
        self.loss.n_time_steps = 5
