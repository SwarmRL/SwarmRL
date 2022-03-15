"""
Loss functions based on Proximal policy optimization.

Notes
-----
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""
import copy
from abc import ABC

import numpy as np
import torch
import torch.nn.functional
from torch.distributions import Categorical

from swarmrl.losses.loss import Loss
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task

torch.autograd.set_detect_anomaly(True)


class ProximalPolicyLoss(Loss, ABC):
    """
    Class to implement the proximal policy loss.
    """

    n_particles: int
    n_time_steps: int

    def __init__(self, n_epochs: int = 10, epsilon: float = 0.2):
        """
        Constructor for the PPO class.

        Parameters
        ----------
        n_epochs : int
                Number of epochs to use in each PPO cycle.
        """
        self.n_epochs = n_epochs
        self.epsilon = epsilon

    def compute_true_value_function(
        self, rewards: list, gamma: float = 0.99, standardize: bool = True
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
        expected_returns : torch.Tensor (n_timesteps, )
                expected returns for each particle
        """
        true_value_function = torch.zeros(self.n_time_steps)
        current_value_state = torch.tensor(0)

        for i in range(self.n_time_steps)[::-1]:
            current_value_state = rewards[i] + current_value_state * gamma

            true_value_function[i] = current_value_state

        # Standardize the value function.
        if standardize:
            mean = torch.mean(torch.tensor(true_value_function), dim=0)
            std = torch.std(torch.tensor(true_value_function), dim=0)

            true_value_function = (true_value_function - mean) / std

        return true_value_function

    def calculate_surrogate_loss(
        self, new_log_probs, old_log_probs, advantage: float, epsilon: float = 0.2
    ):
        """
        Calculates the surrogate loss using the (clamped) ratio * advantage.
        Will be used in compute_loss_values method.

        Parameters
        ----------
        new_log_probs: Float
            Element of a list of the log probabilities at the current step k
        old_log_probs: Float
            Element of a list of the old log probabilities at the previous step
            k-1. instantiated as copy of new_log_probs
        advantage: Float
            Difference between actual return and value function estimates
        epsilon: Float
            Float to specify how much the loss can change in one step. Default is 0.2,
            as it is in an OpenAi paper.

        Returns
        -------
            surrogate_loss : Tensor
        -------

        """
        ratio = torch.exp(new_log_probs - old_log_probs)

        surrogate_1 = ratio * advantage
        surrogate_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        surrogate_loss = -1 * torch.min(surrogate_1, surrogate_2)

        return surrogate_loss

    def compute_loss_values(
        self,
        new_log_probs: list,
        old_log_probs: list,
        new_values: list,
        new_entropy: list,
        rewards: list,
    ):
        """

        Parameters
        ----------
        new_log_probs
        old_log_probs
        new_values
        new_entropy
        rewards

        Returns
        -------
        total loss for the particle to be used in the back-propagation of both the
        actor and the critic.
        """
        true_value_function = self.compute_true_value_function(rewards)
        advantage = true_value_function - torch.tensor(new_values)

        actor_loss = torch.tensor(0, dtype=torch.double)
        critic_loss = torch.tensor(0, dtype=torch.double)

        for i in range(self.n_time_steps):
            surrogate_loss = self.calculate_surrogate_loss(
                new_log_probs[i], old_log_probs[i], advantage[i].item()
            )
            critic_loss = 0.5 * torch.nn.functional.smooth_l1_loss(
                true_value_function[i], new_values[i]
            )
            entropy_loss = -0.01 * new_entropy[i]

            actor_loss += surrogate_loss + critic_loss.item() + entropy_loss
            critic_loss += surrogate_loss.item() + critic_loss + entropy_loss.item()

        return actor_loss, critic_loss

    def compute_actor_values(
        self,
        actor: Network,
        old_actor: Network,
        feature_vector: torch.Tensor,
        log_probs: list,
        old_log_probs: list,
        entropy: list,
    ):
        """
        Takes as input the log_probs, old_log_probs, and entropy values and returns
        the updated list to be used in compute_loss method.

        Parameters
        ----------
        actor: weights of actor NN at new step k
        old_actor: weights of actor NN at old step k-1
        feature_vector
        log_probs: A list of tensors of the log probabilities at the current step k
        old_log_probs: A list of the old log probabilities at the previous step k-1.
        instantiated as copy of new_log_probs entropy

        Returns
        -------
        Updated log_probs,
        old_log_probs,
        entropy
        """
        # Compute old actor values
        old_initial_prob = old_actor(feature_vector)
        old_initial_prob = old_initial_prob / torch.max(old_initial_prob)
        old_action_probability = torch.nn.functional.softmax(old_initial_prob, dim=-1)
        old_distribution = Categorical(old_action_probability)
        old_index = old_distribution.sample()
        old_log_probs.append(old_distribution.log_prob(old_index))

        # Compute actor values
        initial_prob = actor(feature_vector)
        initial_prob = initial_prob / torch.max(initial_prob)
        action_probability = torch.nn.functional.softmax(initial_prob, dim=-1)
        distribution = Categorical(action_probability)
        index = distribution.sample()
        log_probs.append(distribution.log_prob(index))
        entropy.append(distribution.entropy())

        return log_probs, old_log_probs, entropy

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
        print(f"{observable=}")
        print(f"{episode_data=}")
        print(f"{task=}")

        self.n_particles = np.shape(episode_data)[1]
        self.n_time_steps = np.shape(episode_data)[0]

        for _ in range(self.n_epochs):
            old_actor = copy.deepcopy(actor)

            # Actor and critic losses.
            actor_loss = torch.tensor(0, dtype=torch.double)
            critic_loss = torch.tensor(0, dtype=torch.double)

            for i in range(self.n_particles):
                values = []
                log_probs = []
                old_log_probs = []
                entropy = []
                rewards = []
                for j in range(self.n_time_steps):
                    # Compute observable
                    colloid = episode_data[j][i]
                    other_colloids = [c for c in episode_data[j] if c is not colloid]
                    feature_vector = observable.compute_observable(
                        colloid, other_colloids
                    )

                    log_probs, old_log_probs, entropy = self.compute_actor_values(
                        actor,
                        old_actor,
                        feature_vector,
                        log_probs,
                        old_log_probs,
                        entropy,
                    )

                    # Compute critic values
                    values.append(critic(feature_vector))

                    # Compute reward
                    rewards.append(task(feature_vector))

                a_loss, c_loss = self.compute_loss_values(
                    new_log_probs=log_probs,
                    old_log_probs=old_log_probs,
                    new_values=values,
                    new_entropy=entropy,
                    rewards=rewards,
                )
                actor_loss += a_loss
                critic_loss += c_loss

            actor.update_model([actor_loss])
            critic.update_model([critic_loss])

        return actor, critic
