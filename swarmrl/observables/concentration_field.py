"""
Historical position observable computer.

Notes
-----
Observable for sensing changes in some field value, or, the gradient.
"""
import logging
import time
from abc import ABC
from typing import List

import jax
import jax.numpy as np

from swarmrl.agents import Colloid, Swarm, create_swarm
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

        self.vmapped_fn = jax.jit(
            jax.vmap(
                self.compute_single_observable,
                in_axes=(
                    Swarm(
                        pos=0, director=0, id=0, velocity=0, type=0, type_indices=None
                    ),
                    0,
                ),
            )
        )

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
        swarm = create_swarm(colloids)
        partitioned_swarm = swarm.get_species_swarm(self.particle_type)
        self.historic_fields = self.vmapped_fn(partitioned_swarm)

    def compute_single_observable(
        self,
        colloid: Colloid,
    ) -> tuple:
        """
        Compute the observable for a single colloid.

        Parameters
        ----------
        index : int
                Index of the colloid to compute the observable for.
        colloids : List[Colloid]
                List of colloids in the system.
        """
        position = colloid.pos / self.box_length

        current_distance = np.linalg.norm(self.source - position)

        delta = self.decay_fn(current_distance)

        return self.scale_factor * delta

    def compute_observable(self, swarm: Swarm) -> np.ndarray:
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
        # reference_ids = self.get_colloid_indices(colloids)
        start = time.time()
        partitioned_swarm = swarm.get_species_swarm(self.particle_type)
        print(f"A: {time.time() - start}")
        if self._historic_positions == {}:
            msg = (
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )
            raise ValueError(msg)

        fields = self.vmapped_fn(partitioned_swarm)
        observables = fields - self.historic_fields
        self.historic_fields = fields
        return np.array(observables).reshape(-1, 1)
