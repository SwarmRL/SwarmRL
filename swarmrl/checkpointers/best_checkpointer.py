"""
Module for the Save-Best-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.checkpointer import Checkpointer

logger = logging.getLogger(__name__)


class BestRewardCheckpointer(Checkpointer):
    """
    Checkpointer that saves the model if a new best reward is achieved.
    """

    def __init__(
        self, min_reward=250, increase_factor=1.05, window_width=30, wait_time=10
    ):
        super().__init__()
        self.min_reward = min_reward
        self.increase_factor = increase_factor
        self.wait_time = wait_time
        self.window_width = window_width
        self.old_max = 0
        self.next_check_episode = -1

    def check_for_checkpoint(self, rewards: np.ndarray, current_episode: int) -> bool:
        """
        Check if the average reward in the window exceed
        the minimum reward and old max reward.

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
        if current_episode < self.next_check_episode:
            return False

        if current_episode > self.window_width:
            avg_reward = np.mean(
                rewards[current_episode - self.window_width : current_episode + 1]
            )
        else:
            avg_reward = np.mean(rewards[: current_episode + 1])

        if (
            avg_reward > self.increase_factor * self.old_max
            and avg_reward > self.min_reward
        ):
            self.old_max = avg_reward
            self.next_check_episode = current_episode + self.wait_time
            return True
        return False