"""
Module for the intrinsic reward parent class.
"""

from abc import ABC
from dataclasses import dataclass, field

import jax.numpy as np


@dataclass
class TrajectoryInformation:
    """
    Helper dataclass for training RL models.
    """

    particle_type: int
    features: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    killed: bool = False


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
