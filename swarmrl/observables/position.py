"""
Position observable computer.
"""
from abc import ABC
import numpy as np
import torch

from .observable import Observable


class PositionObservable(Observable, ABC):
    """
    Position in box observable.
    """

    def compute_observable(self, colloid: object, other_colloids: list):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.

        Returns
        -------

        """
        data = np.copy(colloid.pos)

        return torch.tensor(data).double()
