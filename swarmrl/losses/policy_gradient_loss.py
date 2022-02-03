"""
Module for the parent class of the loss models.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""
import torch
from typing import Tuple
import torch.nn.functional
import numpy as np
from swarmrl.losses.loss import Loss


class PolicyGradientLoss(Loss):
    """
    Parent class for the reinforcement learning tasks.

    Notes
    -----
    """

    def __init__(self):
        """
        Constructor for the reward class.
        """
        super(Loss, self).__init__()

    def compute_discounted_returns(
            self, rewards: torch.tensor, gamma: float = 0.99, standardize: bool = False
    ):
        """
        Compute the expected returns vector from the tasks.

        Parameters
        ----------
        rewards : torch.Tensor
                A tensor of scalar tasks on which the expected value is computed.
        gamma : float (default=0.99)
                A decay factor for the value of the tasks.
        standardize : bool (default=True)
                If true, the results should be standardized.

        Returns
        -------
        expected_returns : torch.Tensor (n_particles, n_episodes)
                expected returns for each particle
        """
        n_episodes = rewards.shape[1]  # number of episodes.
        expected_returns = torch.empty(
            size=rewards.shape, dtype=torch.float64
        )
        t = torch.linspace(0, n_episodes, n_episodes, dtype=torch.int)
        for i in torch.range(0, n_episodes - 1, dtype=torch.int):
            reward_subset = rewards[:, i:]
            time_subset = t[i:] - torch.ones(n_episodes - i) * i
            expected_returns[:, i] = torch.sum(
                (gamma ** time_subset) * reward_subset, dim=1
            )
        mean = torch.reshape(torch.mean(expected_returns, dim=1), (self.particles, 1))
        std = torch.reshape(torch.std(expected_returns, dim=1), (self.particles, 1))
        if standardize:
            expected_returns = ((expected_returns - mean) /std)

        return expected_returns

    def compute_actor_loss(
            self,
            policy_probabilities: torch.Tensor,
            predicted_rewards: torch.Tensor,
            rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the actor loss.

        Parameters
        ----------
        policy_probabilities : torch.Tensor
                Probabilities returned by the actor.
        predicted_rewards
                Rewards predicted by the critic.
        rewards
                Real rewards.

        Returns
        -------

        """
        expected_returns = self.compute_discounted_returns(rewards)
        advantage = expected_returns - predicted_rewards
        log_probabilities = torch.log(policy_probabilities)

        losses = -1 * torch.sum(log_probabilities * advantage, dim=1)

        return losses

    def compute_critic_loss(
            self, predicted_rewards: torch.Tensor, rewards: torch.Tensor
    ) -> np.ndarray:
        """
        Compute the critic loss.

        Parameters
        ----------
        predicted_rewards : torch.tensor
                Rewards predicted by the critic.
        rewards : torch.tensor
                Real rewards computed by the rewards rule.

        Notes
        -----
        Currently uses the Huber loss.
        """
        huber = torch.nn.HuberLoss()
        expected_returns = self.compute_discounted_returns(rewards)
        loss_vector = np.zeros((self.particles,))
        for i in range(self.particles):
            #loss_vector[i] = huber(predicted_rewards[i], expected_returns[i])
            loss_vector[i] = torch.nn.functional.smooth_l1_loss(
                predicted_rewards[i], expected_returns[i]
            )

        return loss_vector

    def compute_loss(
            self,
            log_probabilities: torch.Tensor,
            values: torch.Tensor,
            rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss functions for the actor and critic based on the reward.

        Parameters
        ----------
        log_probabilities : torch.Tensor (n_particles, n_steps)
                Log of the actions predicted by the actor.
        values : torch.Tensor (n_particles, n_steps)
                Values predicted by the critic.
        rewards : torch.Tensor (n_particles, n_steps)
                Rewards for each state.

        Returns
        -------
        loss_tuple : tuple
                (actor_loss, critic_loss)
        """
        self.particles = log_probabilities.shape[0]

        actor_loss = self.compute_actor_loss(
            log_probabilities, values, rewards
        )
        critic_loss = self.compute_critic_loss(values, rewards)

        return (
            torch.tensor(actor_loss, requires_grad=True),
            torch.tensor(critic_loss, requires_grad=True),
        )
