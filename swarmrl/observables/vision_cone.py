"""
Computes vision cone(s).
"""
from abc import ABC
from typing import List

import jax.numpy as jnp
import numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


class VisionCone(Observable, ABC, Task):
    """
    Shows which particles are within the vision area of the particles.
    """

    def __init__(
        self,
        vision_direction: complex,
        vision_angle: float,
        vision_range: float,
        return_cone: bool,
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        vision_direction: complex
                Direction in which particles are detected. Format: complex(a,b)
                Front: a=0, b=1; back: a=0, b=-1; left: a=-1, b=0; right: a=1,b=0
        vision_angle: float
                Size of the cone angle in degrees.
        vision_range: float
                How far the particles can see.
        return_cone: bool
                If True, the vision cone entails all colloids. If False, it returns only
                the mean distance of the colloids.
        """
        self.vision_angle = vision_angle
        self.vision_range = vision_range
        self.return_cone = return_cone
        self.vision_direction = vision_direction

    def initialize(self, colloids: List[Colloid]):
        """
        This method is not needed for the vision_cone because the initialization
        currently only initialises general features for all colloids while vision cone
        needs to be calculated for each cone separately.
        """
        pass

    def init_task(self):
        """
        When starting, other observables need to have historic data (e.g., the
        concentration field needs at least two time steps). Since the vision cone only
        needs current data, it just returns an empty array to ensure that the NN model
        are initialised with the right input format.
        """
        return jnp.array([], dtype=float)

    def compute_observable(self, colloid: Colloid, other_colloids: List[Colloid]):
        """
        Compute the vision cone of the colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.
        vision_angle: int
                Total angle of view in degrees
        vision_range: float
                How far the particle can see.

        Returns
        List of all particles in the colloids range of vision or
        sum of distances to each particle in range of vision
        """
        vision_half_angle = self.vision_angle / 360 * (np.pi)
        my_pos = jnp.copy(colloid.pos)
        my_director = colloid.director
        my_director = jnp.copy(my_director) + jnp.array(
            [
                jnp.round(self.vision_direction.real, 3),  # x value
                jnp.round(self.vision_direction.imag, 3),  # y value
                0,  # z value
            ]
        )
        # TODO: Add 3D support
        colls_in_vision = []
        colls_distance = []
        for other_p in other_colloids:
            dist = other_p.pos - my_pos
            dist_norm = jnp.linalg.norm(dist)
            in_range = dist_norm < self.vision_range
            if not in_range:
                continue
            in_cone = (
                jnp.arccos(jnp.dot(dist / dist_norm, -my_director)) < vision_half_angle
            )
            if in_cone and in_range:
                colls_in_vision.append(other_p)
                colls_distance.append(dist_norm)
        if self.return_cone is True:
            return colls_in_vision
        else:
            return [jnp.sum(jnp.array(colls_distance))]
