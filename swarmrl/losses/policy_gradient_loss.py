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


def compute_true_value_function(
        rewards: torch.tensor, gamma: float = 0.99, standardize: bool = True
):
    """
    Compute the true value function from the rewards.

    Parameters
    ----------
    rewards : torch.Tensor
            A tensor of scalar tasks on which the expected value is computed.
    gamma : float (default=0.99)
            A decay factor for the value of the tasks.
    standardize : bool (default=True)
            If true, the result is standardized.

    Returns
    -------
    expected_returns : torch.Tensor (n_particles, n_episodes)
            expected returns for each particle
    """
    n_episodes = rewards.shape[1]  # number of episodes.
    n_particles = rewards.shape[0]
    true_value_function = torch.zeros(
        n_particles, n_episodes
    )
    current_value_state = torch.zeros(n_particles)

    for i in range(n_episodes)[::-1]:
        current_value_state = torch.tensor(rewards)[:, i] + current_value_state * gamma

        true_value_function[:, i] = current_value_state

    true_value_function = torch.tensor(true_value_function, requires_grad=True)
    # Standardize the value function.
    if standardize:
        mean = torch.reshape(torch.mean(true_value_function, dim=1), (n_particles, 1))
        std = torch.reshape(torch.std(true_value_function, dim=1), (n_particles, 1))
        true_value_function = ((true_value_function - mean) / std)

    return true_value_function


def compute_critic_loss(
        predicted_rewards: torch.Tensor, rewards: torch.Tensor
) -> torch.Tensor:
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
    n_particles = predicted_rewards.shape[0]
    expected_returns = compute_true_value_function(rewards)
    loss_vector = torch.zeros(n_particles)

    for i in range(n_particles):
        loss_vector[i] = torch.nn.MSELoss(reduction='mean')(
            torch.tensor(predicted_rewards[i]), expected_returns[i]
        )

    return loss_vector


def compute_actor_loss(
        log_probs: torch.Tensor,
        predicted_values: torch.Tensor,
        rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the actor loss.

    Parameters
    ----------
    log_probs : torch.Tensor
            Probabilities returned by the actor.
    predicted_values
            Values predicted by the critic.
    rewards
            Real rewards.

    Returns
    -------

    """
    value_function = compute_true_value_function(rewards)
    advantage = value_function - torch.tensor(predicted_values)

    losses = -1 * torch.sum(torch.tensor(log_probs) * advantage, dim=1)

    return losses


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

    def compute_loss(
            self,
            log_probabilities: torch.Tensor,
            values: torch.Tensor,
            rewards: torch.Tensor,
            entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss functions for the actor and critic based on the reward.

        Returns
        -------
        loss_tuple : tuple
                (actor_loss, critic_loss)
        """
        actor_loss = compute_actor_loss(
            log_probabilities, values, rewards
        )
        critic_loss = compute_critic_loss(values, rewards)

        return actor_loss, critic_loss
