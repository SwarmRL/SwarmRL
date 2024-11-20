"""
Module for the EpisodicTrainer
"""

from typing import TYPE_CHECKING

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.trainers.trainer import Trainer

if TYPE_CHECKING:
    from espressomd import System


class EpisodicTrainer(Trainer):
    """
    Class for the simple MLP RL implementation.

    Attributes
    ----------
    rl_protocols : list(protocol)
            A list of RL protocols to use in the simulation.
    """

    def perform_rl_training(
        self,
        get_engine: callable,
        system: "System",
        n_episodes: int,
        episode_length: int,
        reset_frequency: int = 1,
        load_bar: bool = True,
        save_episodic_data: bool = True,
        save_best_network: str = None,
    ):
        """
        Perform the RL training.

        Parameters
        ----------
        get_engine : callable
                Function to get the engine for the simulation.
        system_runner : espressomd.System
                Engine used to perform steps for each agent.
        n_episodes : int
                Number of episodes to use in the training.
        episode_length : int
                Number of time steps in one episode.
        reset_frequency : int (default=1)
                After how many episodes is the simulation reset.
        load_bar : bool (default=True)
                If true, show a progress bar.
        save_episodic_data : bool (default=False)
                If true, save the episode data. If false, the data is of the
                last episode is overwritten by the new data. Make sure that the
                system runner supports episodic data saving. The get_engine function
                should take a system and a str(cycle_index) as arguments. The
                cycle_index is passed to the EsperessoMD engine as 'h5_group_tag'. See
                the implementationin the test_semi_episodic_data_writing function in
                CI/espresso_tests/integration_tests/test_rl_trainers.py
        save_best_network : str
                Saves the network with the best achieved reward at the given directory.
                This is measured by an average over 10 episodes.

        Notes
        -----
        If you are using semi-episodic training but your task kills the
        simulation, the system will be reset.
        """
        killed = False
        rewards = [0.0]
        current_reward = 0.0
        force_fn = self.initialize_training()
        cycle_index = 0
        best_running_reward = 0.0
        running_reward = 0.0
        progress = Progress(
            "Episode: {task.fields[Episode]}",
            BarColumn(),
            "Episode reward: {task.fields[current_reward]} Running Reward:"
            " {task.fields[running_reward]}",
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task(
                "Episodic Training",
                total=n_episodes,
                Episode=0,
                current_reward=current_reward,
                running_reward=np.mean(rewards),
                visible=load_bar,
            )

            for episode in range(n_episodes):
                # Check if the system should be reset.
                if episode % reset_frequency == 0 or killed:
                    print(f"Resetting the system at episode {episode}")
                    self.engine = None
                    if save_episodic_data:
                        try:
                            self.engine = get_engine(system, f"{cycle_index}")
                            cycle_index += 1
                        except TypeError:
                            raise ValueError(
                                "The system runner does not support episodic data"
                                " saving. Your get_engine function should take a system"
                                " and a str(cycle_index) as arguments. The cycle_index"
                                " is passed to the EsperessoMD engine as"
                                " 'h5_group_tag'."
                            )
                    else:
                        self.engine = get_engine(system)

                    # Initialize the tasks and observables.
                    for agent in self.agents.values():
                        agent.reset_agent(self.engine.colloids)

                self.engine.integrate(episode_length, force_fn)

                force_fn, current_reward, killed = self.update_rl()

                rewards.append(current_reward)

                running_reward = np.round(np.mean(rewards[-10:]), 6)

                if save_best_network is not None and running_reward > (
                    best_running_reward * 0.99
                ):  # Scaling the best reward helps to save later networks more often.
                    #So that 'lucky' encounters get ruled out.
                    best_running_reward = running_reward
                    self.export_models(
                        directory=str(save_best_network) + "/best_network"
                    )

                episode += 1
                progress.update(
                    task,
                    advance=1,
                    Episode=episode,
                    current_reward=np.round(current_reward, 4),
                    running_reward=running_reward,
                )
                self.engine.finalize()

        return np.array(rewards)
