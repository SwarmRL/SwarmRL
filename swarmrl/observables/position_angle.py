"""
Give position and angle.
"""
from abc import ABC

import jax.numpy as np
import numpy as onp

from .observable import Observable


class PositionAngleObservable(Observable, ABC):
    """
    Position in box observable.
    """

    _observable_shape = (3,)

    def __init__(self, box_length: np.ndarray):
        """
        Constructor for the observable.

        Parameters
        ----------
        box_length : np.ndarray
                Length of the box with which to normalize.
        """
        self.box_length = box_length

    def compute_observable(self, colloid: object, other_colloids: list):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.
        """
        data = onp.copy(colloid.pos)
        director = onp.copy(colloid.director)

        data = np.concatenate((np.array(data) / (self.box_length / 2), director))
        print(data)
        return data
