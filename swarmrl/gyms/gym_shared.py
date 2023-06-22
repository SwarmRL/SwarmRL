"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
import os
import time
from typing import List, Tuple

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.engine.engine import Engine
from swarmrl.losses.proximal_policy_loss_shared import SharedProximalPolicyLoss
from swarmrl.models import SharedModel
from swarmrl.rl_protocols.shared_ac import SharedActorCritic


class SharedNetworkGym:
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
        rl_protocols: List[SharedActorCritic],
        ppo_epochs: int = 15,
        loss: SharedProximalPolicyLoss = None,
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
        if loss is None:
            loss = SharedProximalPolicyLoss(n_epochs=ppo_epochs)
        self.loss = loss

        # Add the protocols to an easily accessible internal dict.
        self.rl_protocols = {}
        for protocol in rl_protocols:
            self.rl_protocols[str(protocol.particle_type)] = protocol

        force_models = {}
        observables = {}
        tasks = {}
        actions = {}
        for type_, value in self.rl_protocols.items():
            force_models[type_] = value.network
            observables[type_] = value.observable
            tasks[type_] = value.task
            actions[type_] = value.actions

        self.interaction_model = SharedModel(
            force_models=force_models,
            observables=observables,
            record_traj=True,
            tasks=tasks,
            actions=actions,
        )

    def update_rl(self) -> Tuple[SharedModel, np.ndarray]:
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

        for type_, val in self.rl_protocols.items():
            episode_data = np.load(f".traj_data_{type_}.npy", allow_pickle=True)

            new_reward = np.mean(episode_data.item().get("rewards"))
            if np.isnan(new_reward):
                new_reward = 0.0
            reward += new_reward

            # Compute loss for actor and critic.
            if val.network.kind == "network":
                print("training network")
                self.loss.compute_loss(
                    network=val.network,
                    episode_data=episode_data,
                )
            else:
                pass

            self.interaction_model.force_models[type_] = val.network

        return np.array(reward) / len(self.rl_protocols)

    def reset(self, system_runner):
        system_runner.reset_system()
        for type_, protocol in self.rl_protocols.items():
            try:
                protocol.observable.initialize(system_runner.colloids)
                protocol.task.initialize(system_runner.colloids)
            except AttributeError:
                pass
            try:
                protocol.network.reset()
            except AttributeError:
                pass

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
        for type_, protocol in self.rl_protocols.items():
            try:
                protocol.network.export_model(
                    filename=f"NetworkMode_{type_}", directory=directory
                )
            except AttributeError:
                pass

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
        for type_, protocol in self.rl_protocols.items():
            if protocol.network.kind == "network":
                protocol.network.restore_model_state(
                    filename=f"NetworkMode_{type_}", directory=directory
                )
            else:
                pass

    def initialize_models(self):
        """
        Initialize all of the models in the gym.
        """
        for item, protocol in self.rl_protocols.items():
            protocol.network.reinitialize_network()

    def perform_rl_training(
        self,
        system_runner: Engine,
        n_episodes: int,
        episode_length: int,
        load_bar: bool = True,
        episodic_training: bool = False,
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
        load_bar : bool (default=True)
                Whether to show a progress bar.
        episodic_training : bool (default=False)
                If true, perform episodic training. Otherwise, perform online training.
                If true the system is reset after each episode.
        """
        rewards = [0.0]
        current_reward = 0.0
        episode = 0

        # Initialize the tasks and observables.

        for item, val in self.rl_protocols.items():
            try:
                val.observable.initialize(system_runner.colloids)
                val.task.initialize(system_runner.colloids)
            except AttributeError:
                pass

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
                visible=load_bar,
            )
            for k in range(n_episodes):
                start = time.time()
                system_runner.integrate(episode_length, self.interaction_model)
                end = time.time()
                print(f"Simulation {k} took {end - start} seconds.")
                start = time.time()
                current_reward = self.update_rl()
                end = time.time()
                print(f"Training {k} took {end - start} seconds.")
                rewards.append(current_reward)
                if k % 2 == 0:
                    np.save("rewards.npy", np.array(rewards), allow_pickle=True)
                if k % 50 == 0 and k != 0:
                    self.export_models(f"Models_ep_{k}")
                episode += 1
                progress.update(
                    task,
                    advance=1,
                    Episode=episode,
                    current_reward=np.round(current_reward, 2),
                    running_reward=np.round(np.mean(rewards[-10:]), 2),
                )
                if episodic_training:
                    self.reset(system_runner)

                for item, val in self.rl_protocols.items():
                    os.remove(f".traj_data_{item}.npy")

        system_runner.finalize()

        # Remove the file at the end of the training.
        for type_ in self.rl_protocols:
            try:
                os.remove(f".traj_data_{type_}.npy")
            except FileNotFoundError:
                pass

        return np.array(rewards)
