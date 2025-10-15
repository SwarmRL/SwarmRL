"""
Module for the Save-a-Backup-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer

logger = logging.getLogger(__name__)


class BackupCheckpointer(BaseCheckpointer):
    """
    Checkpointer that saves the current model if the reward starts to decrease.
    This model could be used as a backupt in case of forgetting.
    """

    def __init__(
        self,
        out_path: str,
        min_backup_reward: float = 250,
        window_width: int = 30,
        wait_time: int = 10,
    ):
        """
        Initializes the BackupCheckpointer.

        Parameters:
        -----------
        out_path: str
            Path to the folder where the models should be stored.
        min_backup_reward: int
            The minimum reward required to trigger a backup below
            which no backup is triggered.
        window_width: int
            Determines how many episodes should be considered for
            the running reward average.
        wait_time: int
            A minimum number of episodes to wait for the next backup check.
            Can prevent frequent backups.
        """
        super().__init__(out_path)
        self.min_backup_reward = min_backup_reward
        self.wait_time = wait_time
        self.window_width = window_width
        self.old_max = 0
        self.next_check_episode = -1
        self.last_reward = 0

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
        if current_episode > self.window_width:
            avg_reward = np.mean(
                rewards[current_episode - self.window_width : current_episode + 1]
            )
        else:
            avg_reward = np.mean(rewards[: current_episode + 1])

        if (
            current_episode > self.next_check_episode
            and self.min_backup_reward < avg_reward < np.max(rewards)
            and avg_reward < self.last_reward
        ):
            self.min_backup_reward = avg_reward
            self.next_check_episode = current_episode + self.wait_time

            if current_episode != 0:
                self.last_reward = avg_reward
            return True
        else:
            if current_episode != 0:
                self.last_reward = avg_reward
            return False
