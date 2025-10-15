"""
Module for the Save-Best-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer

logger = logging.getLogger(__name__)


class BestRewardCheckpointer(BaseCheckpointer):
    """
    Checkpointer that saves the model if a new best reward is achieved.

    """

    def __init__(
        self,
        out_path: str,
        min_reward: float = 250,
        increase_factor: float = 1.05,
        window_width: int = 30,
        wait_time: int = 10,
    ):
        """
        Parameters:
        ----------
        out_path: str
            Path to the folder where the models should be stored.
        min_reward : float
            The minimum reward to save a checkpoint.
        increase_factor : float
            The factor by which the average reward must increase
            to trigger a new checkpoint.
        window_width : int
            The width of the window to average the rewards.
        wait_time : int
            The number of episodes to wait before checking for a new checkpoint.

        """
        super().__init__(out_path)
        self.min_reward = min_reward
        self.increase_factor = increase_factor
        self.wait_time = wait_time
        self.window_width = window_width
        self.old_max = -np.inf
        self.next_check_episode = -1

    def check_for_checkpoint(self, rewards: np.ndarray, current_episode: int) -> bool:
        """
        Check if the average reward in the window exceeds
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

        window_end = current_episode + 1
        window_start = max(0, window_end - self.window_width)
        avg_reward = np.mean(rewards[window_start:window_end])

        if (
            avg_reward > self.increase_factor * self.old_max
            and avg_reward > self.min_reward
        ):
            self.old_max = avg_reward
            self.next_check_episode = current_episode + 1 + self.wait_time
            return True
        return False
