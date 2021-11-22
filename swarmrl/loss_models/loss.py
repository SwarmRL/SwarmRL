"""
Module for the parent class of the loss models.
"""
import torch
from typing import Tuple


class Loss(torch.nn.Module):
    """
    Parent class for the reinforcement learning tasks.

    Notes
    -----
    TODO: Fix losses. Currently all particles see all other particles losses. This is due
          to the expected return computed from the historical rewards. This should be
          adjusted to a slice over the same particle in time.
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

    def compute_expected_returns(
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

        Notes
        -----
        TODO: This loss computes expected returns on all particles. Should only compute
        one at a time.

        """
        n_episodes = int(len(rewards) / self.particles)  # number of episodes.
        expected_returns = torch.empty(size=(n_episodes,), dtype=torch.float64)
        t = torch.linspace(0, n_episodes, n_episodes, dtype=torch.int)

        for i in torch.range(0, n_episodes - 1, dtype=torch.int):
            reward_subset = rewards[i:-1:self.particles]
            time_subset = t[i:] - torch.ones(len(reward_subset))
            expected_returns[i] = torch.sum(gamma ** time_subset * reward_subset)

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
        expected_returns = self.compute_expected_returns(rewards)
        advantage = expected_returns - predicted_rewards
        log_probabilities = torch.log(policy_probabilities)

        return -torch.sum(torch.sum(log_probabilities, dim=1) * advantage)

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
        huber = torch.nn.HuberLoss()
        expected_returns = self.compute_expected_returns(rewards)

        return huber(predicted_rewards, expected_returns)

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

        return (torch.tensor(actor_loss, requires_grad=True),
                torch.tensor(critic_loss, requires_grad=True))
