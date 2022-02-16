"""
Task of identifying the origin of a box.

The reward in this case is computed based on the center of mass of the particles along
with their radius of gyration. The idea being that they should form a ring around the
(0, 0, 0) coordinate of the box and that ring should be tightly formed. The radius of
gyration reward may be adjusted such that the colloids form a tighter or wider ring.
"""
from abc import ABC

import numpy as np
import torch

from swarmrl.tasks.task import Task


class FindLocation(Task, ABC):
    """
    Find a location in a box using distances.
    """

    max_distance: float

    def __init__(
        self,
        location: np.ndarray = np.array([0.0, 0.0, 0.0]),
        side_length: np.ndarray = np.array([1.0, 1.0, 1.0]),
    ):
        """
        Constructor for the find origin task.

        Parameters
        ----------
        location : np.ndarray
                The desired origin.
        side_length : np.ndarray
                Box length from which normalization is performed in each dimension.
        """
        super(FindLocation, self).__init__()
        self.location = location
        self.side_length = side_length
        self._compute_max_distance()

    def _compute_max_distance(self):
        """

        Returns
        -------

        """
        reduced_position = self.location / self.side_length
        corner_distances = [
            np.linalg.norm(reduced_position - [1, 1, 0]),
            np.linalg.norm(reduced_position - [1, -1, 0]),
            np.linalg.norm(reduced_position - [-1, 1, 0]),
            np.linalg.norm(reduced_position - [-1, -1, 0]),
        ]
        self.max_distance = max(corner_distances)

    def compute_reciprocal_distance(self, observable: torch.Tensor):
        """
        Compute the reward for the individual particle.

        Parameters
        ----------
        observable : torch.Tensor
                Observables for one particle at one point in time.

        Returns
        -------
        absolute_distance : float
                Scaled distance from the desired location, bound between 0 and 1.
        """
        distance_vector = (observable - self.location) / self.side_length

        absolute_distance = self.max_distance - np.linalg.norm(distance_vector)

        return absolute_distance

    def compute_reward(self, observables: torch.Tensor):
        """
        Compute the reward on the whole group of particles.

        Parameters
        ----------
        observables : torch.Tensor
                Observables collected during the episode.


        Returns
        -------

        """
        reward = self.compute_reciprocal_distance(observables)

        return reward
