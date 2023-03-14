"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
import os
from typing import List, Tuple

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.engine.engine import Engine
from swarmrl.losses.loss import Loss
from swarmrl.models.ml_model import MLModel
from swarmrl.rl_protocols.actor_critic import ActorCritic


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
        rl_protocols: List[ActorCritic],
        loss: Loss,
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
        for item in rl_protocols:
            self.rl_protocols[str(item.particle_type)] = item

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
        for item, value in self.rl_protocols.items():
            force_models[item] = value.actor
            observables[item] = value.observable
            tasks[item] = value.task
            actions[item] = value.actions

        return MLModel(
            models=force_models,
            observables=observables,
            record_traj=True,
            tasks=tasks,
            actions=actions,
        )

    @property
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
        reward = 0.0  # TODO: Separate between species and optimize visualization.

        force_models = {}
        observables = {}
        tasks = {}
        actions = {}
        for item, val in self.rl_protocols.items():
            episode_data = np.load(f".traj_data_{item}.npy", allow_pickle=True)

            reward += np.mean(episode_data.item().get("rewards"))

            # Compute loss for actor and critic.
            actor_grads, critic_grads = self.loss.compute_loss(
                actor=val.actor,
                critic=val.critic,
                episode_data=episode_data,
            )

            val.actor.update_model(actor_grads)
            val.critic.update_model(critic_grads)

            force_models[item] = val.actor
            observables[item] = val.observable
            tasks[item] = val.task
            actions[item] = val.actions

        # Create a new interaction model.
        interaction_model = MLModel(
            models=force_models,
            observables=observables,
            record_traj=True,
            tasks=tasks,
            actions=actions,
        )
        return interaction_model, np.array(reward) / len(self.rl_protocols)

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

        Notes
        -----
        This is super lazy. We should add this to the rl protocol I guess. Same with the
        model restoration.
        """
        for item, val in self.rl_protocols.items():
            val.actor.export_model(filename=f"ActorModel_{item}", directory=directory)
            val.critic.export_model(filename=f"CriticModel_{item}", directory=directory)

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
        for item, val in self.rl_protocols.items():
            val.actor.restore_model_state(
                filename=f"ActorModel_{item}", directory=directory
            )
            val.critic.restore_model_state(
                filename=f"CriticModel_{item}", directory=directory
            )

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
            for item, val in self.rl_protocols.items():
                val.observable.initialize(system_runner.colloids)

        progress = Progress(
            "Episode: {task.fields[Episode]}",
            BarColumn(),
            (
                "Episode reward: {task.fields[current_reward]} Running Reward:"
                " {task.fields[running_reward]}"
            ),
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
                force_fn, current_reward = self.update_rl
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
        for item in self.rl_protocols:
            try:
                os.remove(f".traj_data_{item}.npy")
            except FileNotFoundError:
                pass
