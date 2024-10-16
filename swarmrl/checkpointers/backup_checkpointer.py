"""
Module for the Save-a-Backup-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.checkpointer import Checkpointer

logger = logging.getLogger(__name__)


class BackupCheckpointer(Checkpointer):
    """
    Checkpointer that saves the model if a new best reward is achieved.
    """

    def __init__(self, min_backup_reward=250, window_width=30, wait_time=10):
        super().__init__()
        self.min_backup_reward = min_backup_reward
        self.wait_time = wait_time
        self.window_width = window_width
        self.old_max = 0
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

        current_reward = rewards[current_episode]
        if (
            current_episode > self.next_check_episode
            and self.min_backup_reward < current_reward < np.max(rewards)
        ):
            self.min_backup_reward = current_reward
            self.next_check_episode = current_episode + self.backup_wait_time
            return True
        else:
            return False
