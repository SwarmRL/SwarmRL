"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
import os
from typing import Tuple

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.engine.engine import Engine
from swarmrl.losses.loss import Loss
from swarmrl.models.ml_model import MLModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


class Gym:
    """
    Class for the simple MLP RL implementation.

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
        actor : Network
                A sequential torch model to use as an actor.
        critic : Network
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
        self.actor = actor
        self.critic = critic
        self.task = task
        self.loss = loss
        self.observable = observable
        self.n_particles = n_particles

    def initialize_training(self) -> MLModel:
        """
        Return an initialized interaction model.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to start the simulation with.
        """
        return MLModel(
            model=self.actor,
            observable=self.observable,
            record_traj=True,
            task=self.task,
        )

    def update_rl(self) -> Tuple[MLModel, np.ndarray]:
        """
        Update the RL algorithm.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to use in the next episode.
        reward : np.ndarray
                Current mean episode reward. This is returned for nice progress bars.
        """
        episode_data = np.load(".traj_data.npy", allow_pickle=True)

        reward = np.mean(episode_data.item().get("rewards"))

        # Compute loss for actor and critic.
        self.loss.compute_loss(
            actor=self.actor,
            critic=self.critic,
            episode_data=episode_data,
        )

        # Create a new interaction model.
        interaction_model = MLModel(
            model=self.actor,
            observable=self.observable,
            record_traj=True,
            task=self.task,
        )

        return interaction_model, reward

    def export_models(self, directory: str = "Models"):
        """
        Export the models to the specified directory.

        Parameters
        ----------
        directory : str (default='Models')
                Directory in which to save the models.

        Returns
        -------
        Saves the actor and the critic to the specific directory.
        """
        self.actor.export_model(f"{directory}/Actor")
        self.critic.export_model(f"{directory}/Critic")

    def restore_models(self, directory: str = "Models"):
        """
        Export the models to the specified directory.

        Parameters
        ----------
        directory : str (default='Models')
                Directory from which to load the objects.

        Returns
        -------
        Loads the actor and critic from the specific directory.
        """
        self.actor.restore_model_state(f"./{directory}/Actor")

        self.critic.restore_model_state(f"./{directory}/Critic")

    def perform_rl_training(
        self,
        system_runner: Engine,
        n_episodes: int,
        episode_length: int,
        initialize: bool = False,
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
        initialize : bool (default=False)
                If true, call the initial colloid positions to initialize a task or
                observable.
        """
        rewards = [0.0]
        current_reward = 0.0
        episode = 0
        force_fn = self.initialize_training()

        if initialize:
            self.observable.initialize(system_runner.colloids)

        progress = Progress(
            "Episode: {task.fields[Episode]}",
            BarColumn(),
            "Episode reward: {task.fields[current_reward]} Running Reward:"
            " {task.fields[running_reward]}",
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task(
                "RL Training",
                total=n_episodes,
                Episode=episode,
                current_reward=current_reward,
                running_reward=np.mean(rewards),
            )
            for _ in range(n_episodes):
                system_runner.integrate(episode_length, force_fn)
                force_fn, current_reward = self.update_rl()
                rewards.append(current_reward)
                episode += 1
                progress.update(
                    task,
                    advance=1,
                    Episode=episode,
                    current_reward=np.round(current_reward, 2),
                    running_reward=np.round(np.mean(rewards[-10:]), 2),
                )

        system_runner.finalize()

        # Remove the file at the end of the training.
        try:
            os.remove(".traj_data.npy")
        except FileNotFoundError:
            pass