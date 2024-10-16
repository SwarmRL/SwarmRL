"""
Module for the Save-at-Goal-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.checkpointer import Checkpointer

logger = logging.getLogger(__name__)


class GoalCheckpointer(Checkpointer):
    """
    Checkpointer that saves the model when a reward goal is reached.
    """

    def __init__(
        self,
        required_reward=200,
        window_width=30,
        do_goal_break=False,
        running_out_length=0,
    ):
        super().__init__()
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
        if current_episode > self.window_width:
            avg_reward = np.mean(
                rewards[current_episode - self.window_width : current_episode + 1]
            )
        else:
            avg_reward = np.mean(rewards[: current_episode + 1])

        if self.DO_GOAL_BREAK:
            if avg_reward >= self.required_reward or self.BREAK_TRAINING:
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
