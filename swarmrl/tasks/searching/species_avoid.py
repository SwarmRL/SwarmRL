"""
Class for the species search task.
"""
import logging
from typing import List

import jax.numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task

logger = logging.getLogger(__name__)


class SpeciesAvoid(Task):
    """
    Class for the species search task.
    """

    def __init__(
        self,
        box_length: np.ndarray = np.array([1000.0, 1000.0, 1000.0]),
        sensing_type: int = 0,
        particle_type: int = 1,
        cutoff: float = 0.1,
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        decay_fn : callable
                Decay function of the field.
        box_size : np.ndarray
                Array for scaling of the distances.
        sensing_type : int (default=0)
                Type of particle to sense.
        scale_factor : int (default=100)
                Scaling factor for the observable.
        avoid : bool (default=False)
                Whether to avoid or move to the sensing type.
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)

        self.box_length = box_length
        self.sensing_type = sensing_type
        self.cut_off = cutoff

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
        pass

    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward on the colloids.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of all colloids in the system.

        Returns
        -------
        rewards : List[float] (n_colloids, dimension)
                List of rewards, one for each colloid.
        """

        particle_pos = (
            np.array(
                [
                    colloid.pos
                    for colloid in colloids
                    if colloid.type == self.particle_type
                ]
            )
            / self.box_length
        )

        preditor_pos = (
            np.array(
                [
                    colloid.pos
                    for colloid in colloids
                    if colloid.type == self.sensing_type
                ]
            )
            / self.box_length
        )

        # compute distance between particles and all predators
        dist = np.linalg.norm(
            particle_pos[:, None, :] - preditor_pos[None, :, :], axis=-1
        )

        # compute reward
        rewards = np.any(dist < self.cut_off, axis=-1)
        rewards = np.where(rewards, -1.0, 0.0)
        return rewards
