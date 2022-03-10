"""
Run a unit test on the loss module.
"""
import torch
from torch.distributions import Categorical

import swarmrl as srl
from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss

torch.random.manual_seed(42)


class TestLoss:
    """
    Test the loss functions for RL models.
    """

    @classmethod
    def setup_class(cls) -> None:
        """
        Set up the test cl

        Returns
        -------

        """
        cls.loss = ProximalPolicyLoss()
        cls.loss.n_time_steps = 5
        cls.rewards = [
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([2.0], requires_grad=True),
            torch.tensor([3.0], requires_grad=True),
            torch.tensor([4.0], requires_grad=True),
            torch.tensor([5.0], requires_grad=True)
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

    def test_true_values(self):
        """
        Compute the discounted returns of the actual rewards.

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

    def test_surrogate_loss(self):
        """
        Test the surrogate loss for 1 particle and equal old and new log probs.
        """
        # Update class parameters for new case.
        self.loss.n_particles = 1
        self.loss.n_time_steps = 5

        new_log_probs = torch.log(torch.tensor([0.2, 0.3, 0.05, 0.15, 0.3]))
        old_log_probs = torch.clone(new_log_probs)

        adv = torch.rand(self.loss.n_time_steps)
        expected_loss = -1 * adv
        predictions = []

        for i in range(self.loss.n_time_steps):
            predictions.append(
                self.loss.calculate_surrogate_loss(
                    new_log_probs[i], old_log_probs[i], adv[i].tolist()
                )
            )

        surr_tensor = torch.Tensor(predictions)
        assert torch.allclose(expected_loss, surr_tensor)

        # Reset to defaults for non-linear deployment case.
        self.loss.n_particles = 10
        self.loss.n_time_steps = 5


    def test_gradient_surrogate_loss(self):
        """
        Tests for probabilities with known gradient if the gradient remains as expected.
        """

        new_prob = torch.tensor([1.0], requires_grad=True)
        old_prob = torch.tensor([1.0], requires_grad=True)

        squared_new_probs = torch.pow(new_prob, 2.0)

        surrogate_loss = self.loss.calculate_surrogate_loss(
            new_log_probs=squared_new_probs,
            old_log_probs=old_prob,
            advantage=-1.0,
            epsilon=1000
        )

        surrogate_loss.register_hook(lambda grad: print(grad))
        surrogate_loss.backward(torch.ones_like(surrogate_loss))
        new_prob_gradient, *_ = new_prob.grad.data
        old_prob_gradient, *_ = old_prob.grad.data

        assert torch.allclose(new_prob_gradient, torch.tensor(2, dtype=torch.float))
        assert torch.allclose(old_prob_gradient, torch.tensor(-1, dtype=torch.float))



    def test_compute_actor_values(self):
        """
        Test whether the function keeps the grad.

        Also test the function for a single timestep of one particle.
        Issue: the compute_actor_values function returns only the final logprob and not
        the whole list of probabilities from which it samples. Since sampling is random,
        the test fails. Hence, I did a manual test which passed.

        Returns
        -------

        """
        true_log_probs = []
        feature_vector = torch.tensor([498.4704, 531.5168, 0.6740]).double()

        # Compute true results
        true_initial_prob = self.actor(feature_vector)
        true_initial_prob = true_initial_prob / torch.max(true_initial_prob)
        true_action_prob = torch.nn.functional.softmax(true_initial_prob, dim=-1)
        true_distribution = Categorical(true_action_prob)
        true_index = true_distribution.sample()
        true_log_probs.append(true_distribution.log_prob(true_index))

        # Compute result of function
        log_probs, old_log_probs, entropy = self.loss.compute_actor_values(
            actor=self.actor,
            old_actor=self.actor,
            feature_vector=feature_vector,
            log_probs=[],
            old_log_probs=[],
            entropy=[],
        )
        print(f'{log_probs=}')
        log_probs[0].register_hook(lambda grad: print(grad))
        log_probs[0].backward(torch.ones_like(log_probs[0]))
        gradient, *_ = initial_prob.grad.data
        print(f'{gradient=}')

        assert true_log_probs == log_probs
