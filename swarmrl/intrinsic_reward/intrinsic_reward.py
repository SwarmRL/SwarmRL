"""
Module for the intrinsic reward parent class.
"""

from abc import ABC

import jax.numpy as np

from swarmrl.utils.colloid_utils import TrajectoryInformation


class IntrinsicReward(ABC):
    """
    Parent class for a SwarmRL intrinsic reward.
    """

    def update(self, episode_data: TrajectoryInformation):
        """
        Update the intrinsic reward on a given episode of data.

        Parameters
        ----------
        episode_data : dict
                A dictionary of episode data.
        """
        raise NotImplementedError("Implemented in child class.")

    def compute_reward(self, episode_data: TrajectoryInformation) -> np.ndarray:
        """
        Compute the intrinsic reward.

        Parameters
        ----------
        episode_data : dict
                A dictionary of episode data.
        """
        raise NotImplementedError("Implemented in child class.")
