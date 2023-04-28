"""
Position observable computer.
"""
from abc import ABC
from typing import List

import jax.numpy as np
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable


class PositionObservable(Observable, ABC):
    """
    Position in box observable.
    """

    def __init__(self, box_length: np.ndarray, particle_type: int = 0):
        """
        Constructor for the observable.

        Parameters
        ----------
        box_length : np.ndarray
                Length of the box with which to normalize.
        """
        super().__init__(particle_type=particle_type)
        self.box_length = box_length

    def compute_single_observable(self, index: int, colloids: list):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        index : int
                Index of the colloid for which the observable should be computed.
        colloids : list
                Colloids in the system.
        """
        colloid = colloids[index]

        data = onp.copy(colloid.pos)

        return np.array(data) / self.box_length

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
