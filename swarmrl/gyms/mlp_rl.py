"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
import copy
import os

import torch
import tqdm

from swarmrl.engine.engine import Engine
from swarmrl.losses.loss import Loss
from swarmrl.models.ml_model import MLModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


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

    def initialize_training(self) -> MLModel:
        """
        Return an initialized interaction model.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to start the simulation with.
        """
        return MLModel(model=self.actor, observable=self.observable, record_traj=True)

    def update_rl(self) -> MLModel:
        """
        Update the RL algorithm.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to use in the next episode.
        """
        episode_data = torch.load(".traj_data.pt")

        # Compute loss for actor and critic.
        actor, critic = self.loss.compute_loss(
            actor=copy.deepcopy(self.actor),
            critic=copy.deepcopy(self.critic),
            observable=self.observable,
            episode_data=episode_data,
            task=self.task,
        )
        self.actor = actor
        self.critic = critic

        # Create a new interaction model.
        interaction_model = MLModel(
            model=self.actor, observable=self.observable, record_traj=True
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
            system_runner.integrate(episode_length, force_fn)
            force_fn = self.update_rl()

        system_runner.finalize()

        # Remove the file at the end of the training.
        try:
            os.remove(".traj_data.pt")
        except FileNotFoundError:
            pass
