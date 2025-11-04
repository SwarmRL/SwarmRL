"""
Module for the EpisodicTrainer
"""

from typing import TYPE_CHECKING

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.trainers.trainer import Trainer

if TYPE_CHECKING:
    from espressomd import System

import logging

logger = logging.getLogger(__name__)


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

        Notes
        -----
        If you are using semi-episodic training but your task kills the
        simulation, the system will be reset.
        """
        killed = False
        rewards = np.zeros(n_episodes)
        current_reward = 0.0
        force_fn = self.initialize_training()
        cycle_index = 0
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

            break_training = False
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

                rewards[episode] = current_reward
                if len(self.checkpointers) > 0:
                    export = []
                    save_string = ""
                    for checkpointer in self.checkpointers:
                        export.append(
                            checkpointer.check_for_checkpoint(rewards, episode)
                        )
                        if export[-1]:
                            save_string += f"-{checkpointer.__class__.__name__}"

                    if any(export):
                        self.export_models(
                            f"{self.checkpoint_path}/Model-ep_{episode + 1}"
                            f"-cur_reward_{current_reward:.1f}"
                            f"{save_string}"
                            + "/"
                        )

                logger.debug(f"{episode=}")
                logger.debug(f"{current_reward=}")

                display_episode = episode + 1
                if display_episode < 10:
                    running_reward = np.round(np.mean(rewards[:display_episode]), 2)
                else:
                    running_reward = np.round(
                        np.mean(rewards[display_episode - 10 : display_episode]), 2
                    )

                progress.update(
                    task,
                    advance=1,
                    Episode=episode,
                    current_reward=np.round(current_reward, 2),
                    running_reward=running_reward,
                )
                self.engine.finalize()

                if break_training is False:
                    for checkpointer in self.checkpointers:
                        if checkpointer.check_for_break():
                            break_training = True
                            self.stop_episode = checkpointer.get_stop_episode()
                else:
                    if episode <= self.stop_episode:
                        print(
                            "Stopping criterion reached, but running out training"
                            f" until {self.stop_episode}"
                        )
                    else:
                        print(f"Stopping training at episode {episode}")
                        break

        return np.array(rewards)
