"""
Module for the parent class of the loss models.
"""
import torch
from typing import Tuple


class Loss(torch.nn.Module):
    """
    Parent class for the reinforcement learning tasks.
    """
    def __init__(self):
        """
        Constructor for the reward class.
        """
        super(Loss, self).__init__()

    @staticmethod
    def compute_expected_returns(
            rewards: torch.tensor, gamma: float = 0.99, standardize: bool = True
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

        """
        expected_returns = torch.empty(size=(len(rewards),), dtype=torch.float64)
        t = torch.linspace(0, len(rewards), len(rewards), dtype=torch.int)

        for i in torch.range(0, len(rewards)):
            reward_subset = rewards[i:]
            time_subset = t[i:] - torch.ones(len(reward_subset))
            expected_returns[i] = torch.sum(gamma**time_subset * reward_subset)

        return expected_returns

    def actor_loss(
            self,
            policy_probabilities: torch.Tensor,
            predicted_rewards: torch.Tensor,
            rewards: torch.Tensor
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

        return -torch.sum(log_probabilities * advantage)

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
            rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss functions for the actor and critic based on the reward.

        Returns
        -------

        """
        actor_loss = self.actor_loss(policy_probabilities, predicted_rewards, rewards)
        critic_loss = self.critic_loss(predicted_rewards, rewards)

        return actor_loss, critic_loss
