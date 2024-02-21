"""
Module to implement a simple multi-layer perceptron for the colloids.
"""

import numpy as np
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.engine.engine import Engine
from swarmrl.trainers.trainer import Trainer


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
        rewards = [0.0]
        current_reward = 0.0
        episode = 0
        force_fn = self.initialize_training()
        
        # Initialize the tasks and observables.
        
        # for agent in self.agents.values():
        #     agent.reset_agent(self.engine.colloids)

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
            for _ in range(n_episodes):

                self.engine.integrate(episode_length, force_fn)

                force_fn, current_reward, killed = self.update_rl()

                if killed:
                    print("Simulation has been ended by the task, ending training.")
                    system_runner.finalize()
                    break

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
