"""
Module for the EpisodicTrainer
"""

from typing import List, Tuple

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

# from swarmrl.losses.loss import Loss
# from swarmrl.losses.proximal_policy_loss import ProximalPolicyLoss
from swarmrl.models.ml_model import MLModel
from swarmrl.rl_protocols.actor_critic import ActorCritic


class EpisodicTrainer:
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
        loss: object = None,
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
        This is super lazy. We should add this to the rl protocol. Same with the
        model restoration.
        """
        for type_, val in self.rl_protocols.items():
            val.network.export_model(filename=f"Model{type_}", directory=directory)

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
        for type_, val in self.rl_protocols.items():
            val.network.restore_model_state(
                filename=f"Model{type_}", directory=directory
            )

    def initialize_models(self):
        """
        Initialize all of the models in the gym.
        """
        for _, val in self.rl_protocols.items():
            val.network.reinitialize_network()

    def setup_engine(self):
        """
        Setup the engine for the simulation.
        """

    def perform_rl_training(
        self,
        get_engine: callable,
        system: object,
        n_episodes: int,
        episode_length: int,
        load_bar: bool = True,
    ):
        """
        Perform the RL training.

        Parameters
        ----------
        get_engine : callable
                Function to get the engine for the simulation.
        system_runner : Engine
                Engine used to perform steps for each agent.
        n_episodes : int
                Number of episodes to use in the training.
        episode_length : int
                Number of time steps in one episode.
        load_bar : bool (default=True)
                If true, show a progress bar.
        """
        rewards = [0.0]
        current_reward = 0.0
        episode = 0
        force_fn = self.initialize_training()

        progress = Progress(
            "Episode: {task.fields[Episode]}",
            BarColumn(),
            "Episode reward: {task.fields[current_reward]} Running Reward:"
            " {task.fields[running_reward]}",
            TimeRemainingColumn(),
        )
        state = 0
        with progress:
            task = progress.add_task(
                "Episodic Training",
                total=n_episodes,
                Episode=episode,
                current_reward=current_reward,
                running_reward=np.mean(rewards),
                visible=load_bar,
            )
            for _ in range(n_episodes):

                if state == 0:
                    engine = get_engine(system)
                    engine_1 = None
                    state = 1

                    # Initialize the tasks and observables.
                    for _, val in self.rl_protocols.items():
                        val.observable.initialize(engine.colloids)
                        val.task.initialize(engine.colloids)

                    engine.integrate(episode_length, force_fn)
                else:
                    engine_1 = get_engine(system)
                    engine = None
                    state = 0

                    # Initialize the tasks and observables.
                    for _, val in self.rl_protocols.items():
                        val.observable.initialize(engine_1.colloids)
                        val.task.initialize(engine_1.colloids)

                    engine_1.integrate(episode_length, force_fn)

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

        return np.array(rewards)
