"""
Module for the Save-regularly-Checkpointer
"""

import logging

import numpy as np

from swarmrl.checkpointers.checkpointer import Checkpointer

logger = logging.getLogger(__name__)


class RegularCheckpointer(Checkpointer):
    """
    Checkpointer that saves the model at regular intervals.
    """

    def __init__(self, save_interval=25):
        super().__init__()
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
