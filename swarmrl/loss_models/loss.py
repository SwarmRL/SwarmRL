"""
Module for the parent class of the loss models.
"""
import torch
from typing import Tuple
import numpy as np


class Loss(torch.nn.Module):
    """
    Parent class for the reinforcement learning tasks.

    Notes
    -----
    """

    def __init__(self, n_colloids: int):
        """
        Constructor for the reward class.

        Parameters
        ----------
        n_colloids : int
                Number of colloids in the system.
        """
        super(Loss, self).__init__()
        self.particles = n_colloids

    def compute_discounted_returns(
            self, rewards: torch.tensor, gamma: float = 0.99, standardize: bool = True
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
        n_episodes = int(len(rewards) / self.particles)  # number of episodes.
        expected_returns = torch.empty(
            size=(self.particles, n_episodes), dtype=torch.float64
        )
        t = torch.linspace(0, n_episodes, n_episodes, dtype=torch.int)
        rewards = torch.reshape(rewards, (self.particles, n_episodes))
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

    def actor_loss(
            self,
            policy_probabilities: torch.Tensor,
            predicted_rewards: torch.Tensor,
            rewards: torch.Tensor,
    ):
        """
        Compute the actor loss.

        Parameters
        ----------
        policy_probabilities
        predicted_rewards
        rewards
        """
        n_episodes = int(len(predicted_rewards) / self.particles)  # number of episodes.
        predicted_rewards = torch.reshape(
            predicted_rewards, (self.particles, n_episodes)
        )
        print(f"Predicted rewards: {predicted_rewards}")
        policy_probabilities = torch.reshape(
            policy_probabilities, (self.particles, n_episodes)
        )
        print(f"Policy probabilities: {policy_probabilities}")
        expected_returns = self.compute_discounted_returns(rewards)
        advantage = expected_returns - predicted_rewards
        log_probabilities = torch.log(policy_probabilities)

        distances = 1/rewards

        distances = (
                (distances - torch.mean(distances)) /
                (torch.std(distances))
        )

        #return torch.sum(log_probabilities * advantage, dim=1)

        return [torch.sum(torch.clip(distances, min=0))]

    def critic_loss(self, predicted_rewards: torch.Tensor, rewards: torch.Tensor):
        """
        Compute the critic loss.

        Parameters
        ----------
        predicted_rewards
        rewards

        Notes
        -----
        Currently uses the Huber loss.
        """
        n_episodes = int(len(predicted_rewards) / self.particles)  # number of episodes.
        predicted_rewards = torch.reshape(
            predicted_rewards, (self.particles, n_episodes)
        )

        huber = torch.nn.HuberLoss()
        expected_returns = self.compute_discounted_returns(rewards)
        loss_vector = np.zeros((self.particles,))
        for i in range(self.particles):
            loss_vector[i] = huber(predicted_rewards[i], expected_returns[i])

        loss = torch.norm(predicted_rewards - rewards, dim=1)

        # loss = (
        #         (loss - torch.mean(loss)) /
        #         (torch.std(loss))
        # )

        return loss # loss_vector

    def compute_loss(
            self,
            policy_probabilities: torch.Tensor,
            predicted_rewards: torch.Tensor,
            rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss functions for the actor and critic based on the reward.

        Returns
        -------

        """
        actor_loss = self.actor_loss(policy_probabilities, predicted_rewards, rewards)
        critic_loss = self.critic_loss(predicted_rewards, rewards)

        print(f"Actor loss: {actor_loss}")
        print(f"Critic loss: {critic_loss}")

        return (
            torch.tensor(actor_loss, requires_grad=True),
            torch.tensor(critic_loss, requires_grad=True),
        )
