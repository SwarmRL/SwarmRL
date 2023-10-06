"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
import os
from typing import List, Tuple

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.engine.engine import Engine
from swarmrl.losses.loss import Loss
from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss
from swarmrl.models.ml_model import MLModel
from swarmrl.rl_protocols.actor_critic import ActorCritic


class Gym:
    """
    Class for the simple MLP RL implementation.

    Attributes
    ----------
    rl_protocols : list(protocol)
            A list of RL protocols to use in the simulation.
    loss : Loss
            An optimization method to compute the loss and update the model.
    """

    def __init__(
        self,
        rl_protocols: List[ActorCritic],
        loss: Loss = ProximalPolicyLoss(),
    ):
        """
        Constructor for the MLP RL.

        Parameters
        ----------
        rl_protocols : dict
                A dictionary of RL protocols
        loss : Loss
                A loss model to use in the A-C loss computation.
        """
        self.loss = loss
        self.rl_protocols = {}

        # Add the protocols to an easily accessible internal dict.
        # TODO: Maybe turn into a dataclass? Not sure if it helps yet.
        for protocol in rl_protocols:
            self.rl_protocols[str(protocol.particle_type)] = protocol

    def initialize_training(self) -> MLModel:
        """
        Return an initialized interaction model.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to start the simulation with.
        """
        # Collect the force models for the simulation runs.
        force_models = {}
        observables = {}
        tasks = {}
        actions = {}
        for type_, value in self.rl_protocols.items():
            force_models[type_] = value.network
            observables[type_] = value.observable
            tasks[type_] = value.task
            actions[type_] = value.actions

        return MLModel(
            models=force_models,
            observables=observables,
            record_traj=True,
            tasks=tasks,
            actions=actions,
        )

    def update_rl(self, trajectory_data: dict) -> Tuple[MLModel, np.ndarray]:
        """
        Update the RL algorithm.

        Returns
        -------
        interaction_model : MLModel
                Interaction model to use in the next episode.
        reward : np.ndarray
                Current mean episode reward. This is returned for nice progress bars.
        """
        reward = 0.0  # TODO: Separate between species and optimize visualization.

        force_models = {}
        observables = {}
        tasks = {}
        actions = {}
        for type_, val in self.rl_protocols.items():
            episode_data = trajectory_data[type_]

            reward += np.mean(episode_data.rewards)
            reward += np.mean(episode_data.rewards)

            # Compute loss for actor and critic.
            self.loss.compute_loss(
                network=val.network,
                episode_data=episode_data,
            )

            force_models[type_] = val.network
            observables[type_] = val.observable
            tasks[type_] = val.task
            actions[type_] = val.actions

        # Create a new interaction model.
        interaction_model = MLModel(
            models=force_models,
            observables=observables,
            record_traj=True,
            tasks=tasks,
            actions=actions,
        )
        return interaction_model, np.array(reward) / len(self.rl_protocols)

    def export_models(self, directory: str = "ckpts", episode: int = 0):
        """
        Export the models to the specified directory.

        Parameters
        ----------
        directory : str (default='ckpts')
                Directory in which to save the models.
        episode : int (default=0)
                Episode for which the weights are being saved.

        Returns
        -------
        Saves the actor and the critic to the specific directory.
        """
        for protocol in self.rl_protocols.values():
            protocol.save_model(directory, episode)

    def restore_models(self, directory: str = "ckpts"):
        """
        Export the models to the specified directory.

        Parameters
        ----------
        directory : str (default='ckpts')
                Directory from which to load the objects.

        Returns
        -------
        Loads the actor and critic from the specific directory.
        """
        for protocol in self.rl_protocols.values():
            protocol.restore_parameters(directory=directory)

    def initialize_models(self):
        """
        Initialize all of the models in the gym.
        """
        for protocol in self.rl_protocols.values():
            protocol.network.reinitialize_network()

    def perform_rl_training(
        self,
        system_runner: Engine,
        n_episodes: int,
        episode_length: int,
        load_bar: bool = True,
        callbacks: list = [],
    ) -> np.ndarray:
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
        load_bar : bool (default=True)
                If true, show a progress bar.

        Returns
        -------
        rewards : np.ndarray
                A numpy array of the rewards over all of the
                episodes performed.
        """
        rewards = [0.0]
        current_reward = 0.0
        episode = 0
        force_fn = self.initialize_training()

        # Initialize the tasks and observables.
        for _, val in self.rl_protocols.items():
            val.observable.initialize(system_runner.colloids)
            val.task.initialize(system_runner.colloids)

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
                visible=load_bar,
            )
            for episode in range(n_episodes):
                # start = time.time()
                system_runner.integrate(episode_length, force_fn)
                trajectory_data = force_fn.trajectory_data
                force_fn, current_reward = self.update_rl(
                    trajectory_data=trajectory_data
                )
                rewards.append(current_reward)
                episode += 1
                progress.update(
                    task,
                    advance=1,
                    Episode=episode,
                    current_reward=np.round(current_reward, 2),
                    running_reward=np.round(np.mean(rewards[-10:]), 2),
                )

                # Check the callbacks
                for operation in callbacks:
                    decision, values = operation(episode)
                    if decision:
                        # Store new data.
                        self.export_models(episode=episode)

                        # Remove old data/
                        if values is not None:
                            os.remove(f"ckpts/*_{values}_*")

        system_runner.finalize()

        return np.array(rewards)
