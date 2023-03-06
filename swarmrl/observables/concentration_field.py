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
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable

logger = logging.getLogger(__name__)


class ConcentrationField(Observable, ABC):
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
        source: np.ndarray,
        decay_fn: callable,
        box_length: np.ndarray,
        scale_factor: int = 100,
        particle_type: int = 0,
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        source : np.ndarray
                Source of the field.
        decay_fn : callable
                Decay function of the field.
        box_size : np.ndarray
                Array for scaling of the distances.
        scale_factor : int (default=100)
                Scaling factor for the observable.
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)

        self.source = source / box_length
        self.decay_fn = decay_fn
        self._historic_positions = {}
        self.box_length = box_length
        self.scale_factor = scale_factor
        self._observable_shape = (3,)

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observable with starting positions of the colloids.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Updates the class state.
        """
        for item in colloids:
            index = onp.copy(item.id)
            position = onp.copy(item.pos) / self.box_length
            self._historic_positions[str(index)] = position

    def compute_single_observable(self, index: int, colloids: List[Colloid]) -> float:
        """
        Compute the observable for a single colloid.

        Parameters
        ----------
        index : int
                Index of the colloid to compute the observable for.
        colloids : List[Colloid]
                List of colloids in the system.
        """
        reference_colloid = colloids[index]
        position = onp.copy(reference_colloid.pos) / self.box_length
        index = onp.copy(reference_colloid.id)
        previous_position = self._historic_positions[str(index)]

        # Update historic position.
        self._historic_positions[str(index)] = position

        current_distance = np.linalg.norm((self.source - position))
        historic_distance = np.linalg.norm(self.source - previous_position)

        delta = self.decay_fn(current_distance) - self.decay_fn(historic_distance)

        return self.scale_factor * delta

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
