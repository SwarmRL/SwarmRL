"""
Module for the BaseCheckpointer parent.
"""

import logging

logger = logging.getLogger(__name__)


class BaseCheckpointer:
    """
    A base class for all checkpointers.

    Attributes
    ----------

    """

    def __init__(self, out_path: str):
        """
        Initializes the checkpointer with the specified output path.
        Args:
            out_path str: Path to the folder where the models should be stored.

        """

        self.out_path = out_path

    def check_for_checkpoint(self, *args, **kwargs) -> bool:
        """
        Determine if a checkpoint should be made.

        Returns
        -------
        bool
            Whether or not to save a checkpoint.
        """
        raise NotImplementedError("This method must be implemented in subclass")

    def check_for_break(self, *args, **kwargs) -> bool:
        """
        Determine if the simulation should be stopped.

        Returns
        -------
        bool
            Whether or not the simulation should be ended.
        """
        return False

    def get_stop_episode(self) -> int:
        """
        Get the episode at which training should stop.

        Returns
        -------
        int
            The episode number at which to stop training, or -1 if not set.
        """
        return -1
