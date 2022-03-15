"""
Run a unit test on the loss module.
"""
import torch

import swarmrl as srl
from swarmrl.losses.policy_gradient_loss import PolicyGradientLoss


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
        cls.loss.n_particles = 1
        cls.loss.n_time_steps = 5
        cls.rewards = [
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([2.0], requires_grad=True),
            torch.tensor([3.0], requires_grad=True),
            torch.tensor([4.0], requires_grad=True),
            torch.tensor([5.0], requires_grad=True),
        ]
        cls.actor_stack = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4),
        )
        actor_initialiser = srl.networks.MLP(cls.actor_stack)
        cls.actor = actor_initialiser.double()

        a = [200, 300, 400]
        b = [202, 298, 402]
        c = [200, 296, 404]
        d = [198, 294, 402]
        e = [196, 296, 400]
        cls.feature_vector = torch.Tensor([a, b, c, d, e]).double()

    def test_expected_returns(self):
        """
        Test the true values of the actual returns.

        Notes
        -------
        Checks the actual values return in the discounted returns without
        standardization and with the simple case of gamma=1
        """
        discounted_return = torch.tensor([15, 14, 12, 9, 5])
        value_function = self.loss.compute_true_value_function(
            rewards=self.rewards, gamma=1, standardize=False
        )
        torch.testing.assert_allclose(discounted_return, value_function)

    def test_expected_returns_standardized(self):
        """
        Test the expected return method for standardized data.

        Notes
        -----
        Test that the expected returns are correct.
        """
        discounted_returns = self.loss.compute_true_value_function(rewards=self.rewards)
        torch.testing.assert_allclose(torch.mean(discounted_returns), 0.0)
        torch.testing.assert_allclose(torch.std(discounted_returns), 1.0)

    def test_actor_loss(self):
        """
        Test whether actor loss keeps the logprobs gradient.
        """
        action_probs = [
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([1.0], requires_grad=True),
        ]

        squared_action_probs = []
        for i in range(len(action_probs)):
            squared_action_probs.append(torch.pow(action_probs[i], 2.0))

        predicted_values = self.loss.compute_true_value_function(self.rewards)
        predicted_values = torch.add(predicted_values, 1)

        actor_loss = self.loss.compute_actor_loss(
            log_probs=squared_action_probs,
            rewards=self.rewards,
            predicted_values=predicted_values,
        )
        actor_loss.register_hook(lambda grad: print(grad))
        actor_loss.backward(torch.ones_like(actor_loss))
        gradient, *_ = action_probs[0].grad.data

        assert torch.allclose(gradient, torch.tensor(2, dtype=torch.float))

    def test_critic_loss(self):
        """
        Test the critic loss by giving it the correct predicted rewards.
        """
        returns = self.loss.compute_true_value_function(self.rewards)
        critic_loss = self.loss.compute_critic_loss(
            predicted_rewards=returns, rewards=self.rewards
        )
        assert torch.allclose(critic_loss, torch.tensor(0, dtype=torch.double))
