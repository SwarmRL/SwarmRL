"""
Historical position observable computer.

Notes
-----
Observable for sensing changes in some field value, or, the gradient.
"""
import logging
from abc import ABC
from typing import List

import jax.numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable

logger = logging.getLogger(__name__)


class Colloids(Observable, ABC):
    """
    Position in box observable.

    Attributes
    ----------
    historic_positions : dict
            A dictionary of past positions of the colloid to be used in the gradient
            computation.
    """

    def __init__(
        self,
        particle_type: int = 0,
    ):
        """
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)

    def compute_single_observable(self, index: int, colloids: List[Colloid]) -> float:
        pass

    def compute_observable(self, colloids: List[Colloid]):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.

        Returns
        -------
        observables : List[float] (n_colloids, dimension)
                List of observables, one for each colloid. In this case,
                current field value minus to previous field value.
        """
        reference_ids = self.get_colloid_indices(colloids)

        if self._historic_positions == {}:
            msg = (
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )
            raise ValueError(msg)

        observables = [
            self.compute_single_observable(index, colloids) for index in reference_ids
        ]

        return np.array(observables).reshape(-1, 1)
