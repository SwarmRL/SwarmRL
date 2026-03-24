"""
Module to implement a simple multi-layer perceptron for the colloids.
"""

import logging

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.engine.engine import Engine
from swarmrl.trainers.trainer import Trainer

logger = logging.getLogger(__name__)


class ContinuousTrainer(Trainer):
    """
    Class for the simple MLP RL implementation.

    Attributes
    ----------
    rl_protocols : list(protocol)
            A list of RL protocols to use in the simulation.
    """

    def perform_rl_training(
        self,
        system_runner: Engine,
        n_episodes: int,
        episode_length: int,
        load_bar: bool = True,
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
                If true, show a progress bar.
        """
        self.engine = system_runner
        rewards = np.zeros(n_episodes)
        current_reward = 0.0
        episode = 0
        completed_episodes = 0
        force_fn = self.initialize_training()

        # Initialize the tasks and observables.
        for agent in self.agents.values():
            agent.reset_agent(self.engine.colloids)

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
                running_reward=0.0,
                visible=load_bar,
            )
            for episode in range(n_episodes):
                self.engine.integrate(episode_length, force_fn)
                force_fn, current_reward, killed = self.update_rl()
                rewards[episode] = current_reward
                completed_episodes = episode + 1

                if killed:
                    logger.info(
                        "Simulation has been ended by the task, ending training."
                    )
                    system_runner.finalize()
                    break

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

                running_start = max(0, episode - 9)
                running_reward = np.round(
                    np.mean(rewards[running_start : episode + 1]), 2
                )
                progress.update(
                    task,
                    advance=1,
                    Episode=episode + 1,
                    current_reward=np.round(current_reward, 2),
                    running_reward=running_reward,
                )

        return np.array(rewards[:completed_episodes])
