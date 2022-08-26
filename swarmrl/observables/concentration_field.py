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

    _observable_shape = (3,)

    def __init__(self, source: np.ndarray, decay_fn: callable, box_length: np.ndarray):
        """
        Constructor for the observable.

        Parameters
        ----------
        source : np.ndarray
                Source of the field.
        decay_fn : callable
                Decay function of the field.
        box_length : np.ndarray
                Array for scaling of the distances.
        """
        self.source = source
        self.decay_fn = decay_fn
        self.historic_positions = {}
        self.box_length = box_length

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
            self.historic_positions[str(index)] = position

    def compute_observable(self, colloid: Colloid, other_colloids: List[Colloid]):
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
        delta_field : float
                Current field value minus to previous field value.
        """
        if self.historic_positions == {}:
            msg = (
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )
            raise ValueError(msg)
        position = onp.copy(colloid.pos) / self.box_length
        index = onp.copy(colloid.id)
        previous_position = self.historic_positions[str(index)]

        # Update historic position.
        self.historic_positions[str(index)] = position

        current_distance = np.linalg.norm((self.source - position))
        historic_distance = np.linalg.norm(self.source - previous_position)

        # TODO: make this a real thing and not some arbitrary parameter.
        return 10000 * np.array(
            [self.decay_fn(current_distance) - self.decay_fn(historic_distance)]
        )
