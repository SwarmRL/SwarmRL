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


def compute_episode_len(
    episode_number: int, min_episode_len: int, max_episode_len: int
) -> int:
    """
    Compute the episode length.

    Parameters
    ----------
    episode_number : int
            Current episode number.
    min_episode_len : int
            Minimum episode length.
    max_episode_len : int
            Maximum episode length.

    Returns
    -------
    episode_len : int
            Length of the current episode.
    """
    if episode_number < max_episode_len:
        episode_len = min_episode_len + int(episode_number / 20)
    else:
        episode_len = max_episode_len
    return episode_len


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
        global_task=None,
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
            loss = SharedProximalPolicyLoss(n_epochs=ppo_epochs, epsilon=0.2)
        self.loss = loss
        self.global_task = global_task
        # Add the protocols to an easily accessible internal dict.
        self.rl_protocols = {}
        for protocol in rl_protocols:
            self.rl_protocols[str(protocol.particle_type)] = protocol

    def initialize_training(self) -> SharedModel:
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
            force_models[item] = value.network
            observables[item] = value.observable
            tasks[item] = value.task
            actions[item] = value.actions

        return SharedModel(
            force_models=force_models,
            observables=observables,
            record_traj=True,
            tasks=tasks,
            actions=actions,
            global_task=self.global_task,
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

        force_models = {}
        observables = {}
        tasks = {}
        actions = {}

        for type_, val in self.rl_protocols.items():
            if val.network.kind == "network":
                episode_data = np.load(f".traj_data_{type_}.npy", allow_pickle=True)

                new_reward = np.mean(episode_data.item().get("rewards"))
                if np.isnan(new_reward):
                    new_reward = 0.0
                reward += new_reward

                # Compute loss for actor and critic.
                self.loss.compute_loss(
                    network=val.network,
                    episode_data=episode_data,
                )
            else:
                pass
            force_models[type_] = val.network
            observables[type_] = val.observable
            tasks[type_] = val.task
            actions[type_] = val.actions

        interaction_model = SharedModel(
            force_models=force_models,
            observables=observables,
            record_traj=True,
            tasks=tasks,
            actions=actions,
            global_task=self.global_task,
        )

        return interaction_model, np.array(reward) / len(self.rl_protocols)

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
        scheduler=None,
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

        force_fn = self.initialize_training()

        # Initialize the tasks and observables.
        try:
            self.global_task.initialize(system_runner.colloids)
        except AttributeError:
            pass
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
                if scheduler is not None:
                    dynamic_episode_len = compute_episode_len(
                        episode_number=k,
                        min_episode_len=episode_length,
                        max_episode_len=1000,
                    )
                    system_runner.integrate(dynamic_episode_len, force_fn)
                else:
                    system_runner.integrate(episode_length, force_fn)
                end = time.time()
                print(f"Simulation {k} took {end - start} seconds.")
                start = time.time()
                force_fn, current_reward = self.update_rl()
                end = time.time()
                print(f"Training {k} took {end - start} seconds.")
                rewards.append(current_reward)
                if k % 100 == 0 and k != 0:
                    self.export_models(f"./Models_ep{k}")
                if k % 5 == 0:
                    np.save(
                        "./rewards.npy",
                        np.array(rewards),
                        allow_pickle=True,
                    )
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
                    try:
                        os.remove(f".traj_data_{item}.npy")
                    except FileNotFoundError:
                        pass

        system_runner.finalize()

        # Remove the file at the end of the training.
        for type_ in self.rl_protocols:
            try:
                os.remove(f".traj_data_{type_}.npy")
            except FileNotFoundError:
                pass

        return np.array(rewards)
