"""
Module for the Save-at-Goal-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer

logger = logging.getLogger(__name__)


class GoalCheckpointer(BaseCheckpointer):
    """
    Checkpointer that saves the model when a reward goal is reached.
    """

    def __init__(
        self,
        out_path: str,
        required_reward: float = 200,
        window_width: int = 30,
        do_goal_break: bool = False,
        running_out_length: int = 0,
    ):
        """
        Initializes the GoalCheckpointer.

        Parameters:
        -----------
        out_path: str
            Path to the folder where the models should be stored.
        required_reward: float
            The reward that needs to be achieved to trigger a checkpoint.
        window_width: int
            Determines how many episodes should be considered
            for the running reward average.
        do_goal_break: bool
            Whether to stop training after the goal is reached.
        running_out_length: int
            The number of episodes to run after the goal is reached
            before stopping training.
        """
        super().__init__(out_path)
        self.required_reward = required_reward
        self.window_width = window_width
        self.DO_GOAL_BREAK = do_goal_break
        self.running_out_length = running_out_length
        self.BREAK_TRAINING = False
        self.stop_episode = -1

    def check_for_checkpoint(self, rewards: np.ndarray, current_episode: int) -> bool:
        """
        Check if the average reward in the window is greater than the required reward.

        Parameters
        ----------
        rewards : np.ndarray
            Array of rewards.
        current_episode : int
            The current episode number.

        Returns
        -------
        bool
            Whether the checkpoint criteria are met.
        """
        window_end = current_episode + 1
        window_start = max(0, window_end - self.window_width)
        avg_reward = np.mean(rewards[window_start:window_end])

        if self.DO_GOAL_BREAK:
            if avg_reward >= self.required_reward and not self.BREAK_TRAINING:
                self.BREAK_TRAINING = True
                if self.running_out_length > 0:
                    self.stop_episode = current_episode + self.running_out_length
                else:
                    self.stop_episode = current_episode
        return avg_reward >= self.required_reward

    def check_for_break(self):
        return self.BREAK_TRAINING

    def get_stop_episode(self):
        return self.stop_episode
