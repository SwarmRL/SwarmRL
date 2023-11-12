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

    def __call__(self, episode: int, **kwargs) -> tuple:
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
            self._n_saves += 1
            if self._n_saves % (self.history) == 0 and self._n_saves != 0:
                removal_index = (
                    self._last_save
                    - (self.history * self.save_interval)
                    + self.save_interval
                )
            else:
                removal_index = None

            return True, removal_index
        else:
            return False, None


@dataclass
class MaxRewardCheckpointing:
    """
    Checkpoint when rewards are larger than previous ones.

    Attributes
    ----------
    test_interval : int
            How often to check the rewards for a better model.
            This will avoid lots of writing related to
            fluctuations from noise.
    """

    test_interval: int

    _last_save: int = 0
    _last_save_episode: int = 0

    def __call__(self, episode: int, reward: float) -> tuple:
        """
        Determine whether or not to checkpoint the models.

        Parameters
        ----------
        episode : int
                Current episode of the model.
        reward : float
                Current episode reward.

        Returns
        -------
        save_decision : bool
                If true, save the parameters
        removal_index : int
                Index of previous files to remove.
        """
        if episode % self.test_interval == 0:
            if reward > self._last_save:
                self._last_save = reward

                remove_index = self._last_save_episode
                self._last_save_episode = episode

                return True, remove_index
            else:
                return False, None
