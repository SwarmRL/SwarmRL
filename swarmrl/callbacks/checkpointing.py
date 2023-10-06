"""
Module for the checkpointing callbacks.
"""
from dataclasses import dataclass


@dataclass
class UniformCheckpointing:
    """
    Perform uniform checkpointing.

    Save model parameters every nth step and keep
    the last m sets of parameters.

    Attributes
    ----------
    save_interval : int (default = 1)
            How often to save the parameters.
    history : int (default = 5)
            How many older files to keep.
    """

    save_interval: int = 1
    history: int = 5

    _last_save: int = 0
    _n_saves: int = 0

    def __call__(self, episode: int) -> tuple:
        """
        Determine whether or not to checkpoint the models.

        Parameters
        ----------
        episode : int
                Current episode of the model.

        Returns
        -------
        save_decision : bool
                If true, save the parameters
        removal_index : int
                Index of previous files to remove.
        """
        if int(episode - self._last_save) >= self.save_interval:
            self._last_save = episode
            if self._n_saves % (self.history + 1) == 0:
                removal_index = self._last_save - self.history
            else:
                removal_index = None

            return True, removal_index
