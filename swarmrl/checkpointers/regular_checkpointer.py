"""
Module for the Save-regularly-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.base_checkpointer import BaseCheckpointer

logger = logging.getLogger(__name__)


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
        super().__init__(out_path)
        self.save_interval = save_interval

    def check_for_checkpoint(self, rewards: np.ndarray, current_episode: int) -> bool:
        """
        Check if the current episode is a multiple of the save interval.

        Parameters
        ----------
        current_episode : int
            The current episode number.

        Returns
        -------
        bool
            Whether the checkpoint criteria are met.
        """
        return (current_episode + 1) % self.save_interval == 0
