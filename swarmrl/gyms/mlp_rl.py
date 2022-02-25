"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
from typing import Tuple

import torch

import tqdm

from swarmrl.losses.loss import Loss
from swarmrl.models.ml_model import MLModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.engine.engine import Engine

import h5py as hf

import numpy as np


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
        n_particles: int,
        database_path: str
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
        self.database_path = database_path

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
        self.critic.update_model(loss, retain=True)

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
        self.actor.update_model(loss, retain=True)

    def _format_episode_data(self, episode_data: list) -> Tuple:
        """
        Format the episode data to use in the training.

        Parameters
        ----------
        episode_data : np.ndarray (n_particles * n_time_steps, 3)
                Data collected during the episode.

        Returns
        -------
        log_prob : torch.Tensor (n_time_steps, n_particles)
                All log probabilities collected during an episode.
        values : torch.Tensor (n_time_steps, n_particles)
                All values collected during the episode.
        rewards : torch.Tensor (n_time_steps, n_particles)
                All rewards collected during the episode.
        entropy : torch.Tensor (n_time_steps, n_particles)
                Distribution entropy computed during the run.

        Notes
        -----
        This is a gradient preserving method, that is, if a tensor requiring a gradient
        comes in here, it is returned also requiring a gradient. DO NOT TOUCH this
        method unless you know very well what you are doing. If the gradients are not
        preserved here, the models will NOT train nor will they give you an error.
        """
        time_steps = int(len(episode_data) / self.n_particles)

        log_probs = []
        values = []
        rewards = []
        entropy = []

        for i in range(time_steps):
            log_probs_snapshot = []
            values_snapshot = []
            rewards_snapshot = []
            entropy_snapshot = []
            for j in range(self.n_particles):
                log_probs_snapshot.append(episode_data[self.n_particles * i + j][0])
                values_snapshot.append(episode_data[self.n_particles * i + j][1])
                rewards_snapshot.append(episode_data[self.n_particles * i + j][2])
                entropy_snapshot.append(episode_data[self.n_particles * i + j][3])

            log_probs.append(log_probs_snapshot)
            values.append(values_snapshot)
            rewards.append(rewards_snapshot)
            entropy.append(entropy_snapshot)

        # Ensure that gradients have been kept
        if not log_probs[0][0].requires_grad:
            err_msg = (
                "WARNING: The values predicted by the actor appear to have lost"
                " their gradient. Without this gradient, the networks will NOT"
                " train. If this was intentional, please ignore this message, if"
                " not, check to see if you have re-cast anything coming out of a"
                " network."
            )
            print(err_msg)
        if not values[0][0].requires_grad:
            err_msg = (
                "WARNING: The values predicted by the critic appear to have lost"
                " their gradient. Without this gradient, the networks will NOT"
                " train. If this was intentional, please ignore this message, if"
                " not, check to see if you have re-cast anything coming out of a"
                " network."
            )
            print(err_msg)

        return log_probs, values, rewards, entropy, time_steps

    def load_last_episode(self, episode_length: int):
        """
        Load the data of the last episode.

        Returns
        -------

        """
        with hf.File(f"{self.database_path}/trajectory.hdf5") as db:
            data = db['colloids']['Unwrapped_Positions'][-episode_length:]

        return data

    def initialize_training(self) -> MLModel:
        """
        Return an initialized interaction model.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to start the simulation with.
        """
        return MLModel(
            gym=self,
            observable=self.observable,
            reward_cls=self.task,
            record=True,
        )

    def update_rl(self, interaction_model: MLModel, episode_length: int, data) -> MLModel:
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
        episode_data = interaction_model.recorded_values
        episode_pos_data = self.load_last_episode(episode_length)
        print(np.shape(data))

        log_prob, values, rewards, entropy, time_steps = self._format_episode_data(
            episode_data
        )

        # Compute loss for actor and critic.
        actor_loss, critic_loss = self.loss.compute_loss(
            log_probabilities=log_prob,
            values=values,
            rewards=rewards,
            entropy=entropy,
            n_particles=self.n_particles,
            n_time_steps=time_steps,
        )

        if not actor_loss[0].requires_grad:
            msg = (
                "Actor loss values do not have an associated gradient. This means"
                "that the networks will not train. Please check if you have "
                "cast any tensor coming from a network as this destroys gradients."
            )
            print(msg)
        if not critic_loss[0].requires_grad:
            msg = (
                "Critic loss values do not have an associated gradient. This means"
                "that the networks will not train. Please check if you have "
                "cast any tensor coming from a network as this destroys gradients."
            )
            print(msg)

        # Perform back-propagation.
        self.update_actor(actor_loss)

        self.update_critic(critic_loss)

        # Create a new interaction model.
        interaction_model = MLModel(
            gym=self,
            observable=self.observable,
            reward_cls=self.task,
            record=True,
        )

        return interaction_model

    def perform_rl_training(
            self, system_runner: Engine, n_episodes: int, episode_length: int
    ):
        """
        Perform the RL training.

        Parameters
        ----------
        system_runner : Engine
                Engine used to perform steps for each agent.
        n_episodes : int
                Number of episodes to use in the training.
        episode_length : int
                Number of time steps in one episode.
        """
        force_fn = self.initialize_training()
        for _ in tqdm.tqdm(range(n_episodes)):
            data = system_runner.integrate(episode_length, force_fn)
            force_fn = self.update_rl(force_fn, episode_length=episode_length, data=data)

        system_runner.finalize()
