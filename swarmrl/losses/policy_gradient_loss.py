"""
Module for the parent class of the loss models.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""
from typing import List

import numpy as np
import torch
import torch.nn.functional
from torch.distributions import Categorical

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


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
        self.n_particles = None
        self.n_time_steps = None

    def compute_loss(
        self,
        actor: Network,
        critic: Network,
        observable: Observable,
        episode_data: list,
        task: Task,
    ):
        """
        Compute the loss functions for the actor and critic based on the reward.

        For full doc string, see the parent class.

        Returns
        -------
        loss_tuple : tuple
                (actor_loss, critic_loss)
        """
        self.n_particles = np.shape(episode_data)[1]
        self.n_time_steps = np.shape(episode_data)[0]

        # Actor and critic losses.
        actor_loss = torch.tensor(0)
        critic_loss = torch.tensor(0)

        for i in range(self.n_particles):
            values = []
            log_probs = []
            rewards = []
            for j in range(self.n_time_steps):
                # Compute observable
                colloid = episode_data[j][i]
                other_colloids = [c for c in episode_data[j] if c is not colloid]
                observable = observable.compute_observable(colloid, other_colloids)

                # Compute actor values
                action_probability = torch.nn.functional.softmax(
                    actor(observable), dim=-1
                )
                distribution = Categorical(action_probability)
                index = distribution.sample()
                log_probs.append(distribution.log_prob(index))

                entropy = distribution.entropy()
                print(entropy)

                # Compute critic values
                values.append(critic(observable))

                # Compute reward
                rewards.append(task(observable))

            actor_loss += self.compute_actor_loss(log_probs, values, rewards)
            critic_loss += self.compute_critic_loss(values, rewards)

    def compute_true_value_function(
        self, rewards: List, gamma: float = 0.99, standardize: bool = True
    ):
        """
        Compute the true value function from the rewards.

        Parameters
        ----------
        rewards : List
                A tensor of scalar tasks on which the expected value is computed.
        gamma : float (default=0.99)
                A decay factor for the value of the tasks.
        standardize : bool (default=True)
                If true, the result is standardized.

        Returns
        -------
        expected_returns : torch.Tensor (n_timesteps, n_particles)
                expected returns for each particle
        """
        true_value_function = torch.zeros(self.n_time_steps, self.n_particles)
        current_value_state = torch.zeros(self.n_particles)

        for i in range(self.n_time_steps)[::-1]:
            current_value_state = (
                torch.tensor(rewards)[i, :] + current_value_state * gamma
            )

            true_value_function[i, :] = current_value_state

        # Standardize the value function.
        if standardize:
            mean = torch.mean(true_value_function, dim=0)
            std = torch.std(true_value_function, dim=0)

            true_value_function = (true_value_function - mean) / std

        return true_value_function

    def compute_critic_loss(self, predicted_rewards: List, rewards: List) -> List:
        """
        Compute the critic loss.

        Parameters
        ----------
        predicted_rewards : List
                Rewards predicted by the critic.
        rewards : List
                Real rewards computed by the rewards rule.

        Notes
        -----
        Currently uses the Huber loss.
        """
        value_function = self.compute_true_value_function(rewards)
        loss_vector = []

        for i in range(self.n_particles):
            for j in range(self.n_time_steps):
                if j == 0:
                    particle_loss = torch.nn.functional.smooth_l1_loss(
                        predicted_rewards[j][i], value_function[j][i]
                    )
                else:
                    particle_loss = particle_loss + torch.nn.functional.smooth_l1_loss(
                        predicted_rewards[j][i], value_function[j][i]
                    )

            loss_vector.append(particle_loss)

        return loss_vector

    def compute_actor_loss(
        self,
        log_probs: List,
        predicted_values: List,
        rewards: List,
    ) -> List:
        """
        Compute the actor loss.

        Parameters
        ----------
        log_probs : List
                Probabilities returned by the actor.
        predicted_values : List
                Values predicted by the critic.
        rewards : List
                Real rewards.

        Returns
        -------
        losses : List
        """
        value_function = self.compute_true_value_function(rewards)
        advantage = value_function - torch.tensor(predicted_values)

        losses = []
        particle_loss = torch.tensor(0)
        for i in range(self.n_time_steps):
            particle_loss = particle_loss + log_probs[i] * advantage[i]

        losses.append(-1 * particle_loss)

        return losses
