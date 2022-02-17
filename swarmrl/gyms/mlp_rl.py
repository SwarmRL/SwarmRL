"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
from typing import Tuple

from swarmrl.models.interaction_model import InteractionModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.losses.loss import Loss
from swarmrl.models.ml_model import MLModel
import numpy as np
import torch


class MLPRL:
    """
    Class for the simple MLP RL implementation.

    The multi-layer perceptron learner is a simple global network model wherein all
    particles are passed into the network and forces computed for each individually.

    Attributes
    ----------
    actor : Network
                A sequential torch model to use as an actor.
    critic : Network
                A sequential torch model to use as a critic.
    task : callable
                Callable function from which a reward can be computed.
    """

    def __init__(
        self,
        actor: Network,
        critic: Network,
        task: Task,
        loss: Loss,
        observable: Observable,
        n_particles: int
    ):
        """
        Constructor for the MLP RL.

        Parameters
        ----------
        actor : torch.nn.Sequential
                A sequential torch model to use as an actor.
        critic : torch.nn.Sequential
                A sequential torch model to use as a critic.
        task : Task
                Callable from which a reward can be computed.
        loss : Loss
                A loss model to use in the A-C loss computation.
        observable : Observable
                Observable class with which to compute the input vector.
        n_particles : int
                Number of particles in the environment.
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.task = task
        self.loss = loss
        self.observable = observable
        self.n_particles = n_particles

    def update_critic(self, loss: torch.Tensor):
        """
        Perform an update on the critic network.

        Parameters
        ----------
        loss : torch.Tensor
                A loss vector to pass to the model.

        Returns
        -------
        Runs back-propagation on the critic model.
        """
        self.critic.update_model(loss)

    def update_actor(self, loss: torch.Tensor):
        """
        Perform an update on the actor network.

        Parameters
        ----------
        loss : torch.Tensor
                A loss vector to pass to the model.
        Returns
        -------
        Runs back-propagation on the actor model.
        """
        self.actor.update_model(loss)

    def _format_episode_data(self, episode_data: torch.Tensor) -> Tuple:
        """
        Format the episode data to use in the training.

        Parameters
        ----------
        episode_data : np.ndarray (n_particles * n_time_steps, 3)
                Data collected during the episode.

        Returns
        -------
        log_prob : torch.Tensor (n_particles, n_time_steps)
                All log probabilities collected during an episode.
        values : torch.Tensor (n_particles, n_time_steps)
                All values collected during the episode.
        rewards : torch.Tensor (n_particles, n_time_steps)
                All rewards collected during the episode.
        entropy : torch.Tensor (n_particles, n_time_steps)
                Distirbution entropy computed during the run.
        """
        time_steps = int(episode_data.shape[0] / self.n_particles)

        concat_log_probs = torch.tensor(episode_data[:, 0])
        concat_values = torch.tensor(episode_data[:, 1])
        concat_rewards = torch.tensor(episode_data[:, 2])
        concat_entropy = torch.tensor(episode_data[:, 3])

        log_probs = np.zeros((self.n_particles, time_steps))
        values = np.zeros((self.n_particles, time_steps))
        rewards = np.zeros((self.n_particles, time_steps))
        entropy = np.zeros((self.n_particles, time_steps))

        for i in range(self.n_particles):
            log_probs[i] = concat_log_probs[i::self.n_particles]
            values[i] = concat_values[i::self.n_particles]
            rewards[i] = concat_rewards[i::self.n_particles]
            entropy[i] = concat_entropy[i::self.n_particles]

        return log_probs, values, rewards, entropy

    def initialize_training(self) -> MLModel:
        """
        Return an initialized interaction model.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to start the simulation with.
        """
        return MLModel(
            actor=self.actor.model,
            critic=self.critic.model,
            observable=self.observable,
            reward_cls=self.task,
            record=True
        )

    def update_rl(self, interaction_model: MLModel) -> MLModel:
        """
        Update the RL algorithm.

        Parameters
        ----------
        interaction_model : MLModel
                Interaction model to read the actor/critic and reward values.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to use in the next episode.
        """
        episode_data = torch.tensor(interaction_model.recorded_values)

        log_prob, values, rewards, entropy = self._format_episode_data(episode_data)

        # Compute loss for actor and critic.
        actor_loss, critic_loss = self.loss.compute_loss(
            log_probabilities=log_prob,
            values=values,
            rewards=rewards,
            entropy=entropy
        )

        # Perform back-propagation.
        self.update_actor(actor_loss)
        self.update_critic(critic_loss)

        # Create a new interaction model.
        interaction_model = MLModel(
            actor=self.actor.model,
            critic=self.critic.model,
            observable=self.observable,
            reward_cls=self.task,
            record=True
        )

        return interaction_model
