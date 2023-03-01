"""
Give position and angle.
"""
from abc import ABC
from typing import List

import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable


class Director(Observable, ABC):
    """
    Position in box observable.
    """

    def __init__(self, particle_type: int = 0):
        """
        Constructor for the observable.

        Parameters
        ----------
        box_length : np.ndarray
                Length of the box with which to normalize.
        """
        super().__init__(particle_type=particle_type)

    def compute_single_observable(self, index: int, colloids: list):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        index : int
                Index of the colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.
        """
        colloid = colloids[index]

        director = onp.copy(colloid.director)

        return director

    def compute_observable(self, colloids: List[Colloid]):
        """
        Compute the current state observable for all colloids.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.
        """
        indices = self.get_colloid_indices(colloids)

        return [self.compute_single_observable(i, colloids) for i in indices]
