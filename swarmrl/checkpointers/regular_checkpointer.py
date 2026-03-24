"""
Module for the Save-regularly-Checkpointer
"""

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer


class RegularCheckpointer(BaseCheckpointer):
    """
    Checkpointer that saves the model at regular intervals.
    """

    def __init__(self, out_path: str, save_interval: int = 25):
        """
        Initializes the RegularCheckpointer.

        Parameters:
        -----------
        save_interval: int
            The interval at which to save the model.
        """
        if save_interval <= 0:
            raise ValueError("save_interval must be greater than 0")

        super().__init__(out_path)
        self.save_interval = save_interval

    def check_for_checkpoint(self, rewards: np.ndarray, current_episode: int) -> bool:
        """
        Check if the current episode is a multiple of the save interval.

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
        if rewards is None or len(rewards) == 0:
            return False
        if current_episode < 0 or current_episode >= len(rewards):
            return False

        return (current_episode + 1) % self.save_interval == 0
