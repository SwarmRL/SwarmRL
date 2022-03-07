"""
Run a unit test on the loss module.
"""
import pytest
import torch
import swarmrl as srl
import copy
import numpy as np

from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss
from torch.distributions import Categorical

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
        rewards = torch.tensor([1,2,3,4,5])
        discounted_return = torch.tensor([15,14,12,9,5])
        value_function = self.loss.compute_true_value_function(
            rewards=rewards, gamma=1, standardize=False
        )
        torch.testing.assert_allclose(discounted_return, value_function)

    def test_expected_returns_standardized(self):
        """
        Test the expected return method for standardized data.

        Notes
        -----
        Test that the expected returns are correct.
        """
        rewards = torch.tensor([1, 2, 3, 4, 5])
        discounted_returns = self.loss.compute_true_value_function(rewards=rewards)
        torch.testing.assert_allclose(torch.mean(discounted_returns).numpy(), 0.0)
        torch.testing.assert_allclose(torch.std(discounted_returns).numpy(), 1.0)

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
            predictions.append(self.loss.calculate_surrogate_loss(
            new_log_probs[i],
            old_log_probs[i],
            adv[i].tolist()
        ))

        surr_tensor = torch.Tensor(predictions)
        assert torch.allclose(expected_loss,surr_tensor)

        # Reset to defaults for non-linear deployment case.
        self.loss.n_particles = 10
        self.loss.n_time_steps = 5

    #TODO write test for compute_loss_values

    def test_compute_actor_values(self):
        """
        Test the function for a single timestep of one particle.
        Returns
        -------

        """
        true_log_probs = []
        feature_vector = torch.tensor([498.4704, 531.5168, 0.6740]).double()

        # Compute true results
        true_initial_prob = self.actor(feature_vector)
        print(f'Result: {true_initial_prob=}')
        true_initial_prob = true_initial_prob / torch.max(true_initial_prob)
        print(f'Result: {true_initial_prob=}')
        true_action_prob = torch.nn.functional.softmax(true_initial_prob, dim=-1)
        print(f'Result: {true_action_prob=}')
        true_distribution = Categorical(true_action_prob)
        true_index = true_distribution.sample()
        print(f'Result: {true_index=}')
        true_log_probs.append(true_distribution.log_prob(true_index))
        print(f'Result: {true_log_probs=}')

        # Compute result of function
        computed_log_probs,computed_old_log_probs,computed_entropy = self.loss.\
            compute_actor_values(
            actor=self.actor,
            old_actor=self.actor,
            feature_vector=feature_vector,
            log_probs= [],
            old_log_probs=[],
            entropy=[]
        )
        print(f'{true_log_probs=}')
        print(f'{computed_log_probs=}')
        print(f'{computed_old_log_probs=}')
        assert torch.allclose(true_log_probs[0], computed_log_probs[0])


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
