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
    The vision cone acts like a camera for the particles. It can either output all
    colloids within its view, a function of the distances between one colloid and all
    other colloids, or the normalised distance to the source.
    """

    def __init__(
        self,
        vision_direction: complex,
        vision_angle: float,
        vision_range: float,
        return_cone: bool,
        source: jnp.ndarray,
        box_size: np.ndarray,
        detect_source: bool = True,
        particle_type: int = 0,
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        vision_direction : complex
                Direction in which particles are detected. Format: complex(a,b)
                Front: a=0, b=1; back: a=0, b=-1; left: a=-1, b=0; right: a=1,b=0:
        vision_angle : float
                Size of the cone angle in degrees.
        vision_range : float
                How far the particles can see.
        return_cone : bool
                If True, the vision cone entails all colloids. If False, it returns only
                the mean distance of the colloids.
        source : jnp.ndarray
                Target which the particles shall reach.
        box_size : np.ndarray
                Array for scaling of the distances.
        detect_source : bool
                If True, the function returns [0.01] if the source is within the vision
                cone to get a large reward.
        particle_type : int (default: 0)
                Type of the particle. If the particle type is not 0.
        """
        self.vision_angle = vision_angle
        self.vision_range = vision_range
        self.return_cone = return_cone
        self.vision_direction = vision_direction
        self.source = source
        self.detect_source = detect_source
        self.box_size = box_size
        self.vision_half_angle = vision_angle / 360 * (np.pi)  # angular view

    def initialize(self, colloids: List[Colloid]):
        """
        This method is not needed for the vision_cone because the initialization
        currently only initialises general features for all colloids while the vision
        cone needs to be calculated for each cone separately.
        """
        pass

    def _calculate_director(self, colloid):
        """
        Calculates the normalised director of the vision cone.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.

        Returns
        Colloid position and vision cone director.
        """
        my_pos = jnp.copy(colloid.pos)
        my_director = jnp.copy(colloid.director)
        my_director = my_director + jnp.array(
            [
                jnp.round(self.vision_direction.real, 3),  # x value
                jnp.round(self.vision_direction.imag, 3),  # y value
                0,  # z value
            ]
        )  # specifies direction of vision cone relative to particle direction
        # TODO: Add 3D support
        my_director = my_director / jnp.linalg.norm(my_director)
        return my_pos, my_director

    def _detect_source(self, my_pos, my_director):
        """
        Calculates the distance from the particle to the source and returns the
        normalised distance if the source is within the vision cone.

        Parameters
        ----------
        my_pos : jnp.ndarray
                Position of the colloid.
        my_director : jnp.ndarray
                Normalised director of the colloid.
        Returns
        Normalised distance to source if source is within the vision cone. Else 0.0.
        """
        dist = my_pos - self.source
        dist_norm = jnp.linalg.norm(dist)
        in_range = dist_norm < self.vision_range
        if in_range:
            in_cone = (
                jnp.arccos(jnp.dot(dist / dist_norm, my_director))
                < self.vision_half_angle
            )
        if in_cone and in_range:
            dist = dist / self.box_size
            return jnp.array([jnp.linalg.norm(dist)])
        else:
            return jnp.array([0.0])

    def _calculate_cone(self, my_pos, my_director, other_colloids):
        """
        Calculates the vision cone of the colloid.

        Parameters
        ----------
        my_pos : jnp.ndarray
                Position of the colloid.
        my_director : jnp.ndarray
                Normalised director of the colloid.
        other_colloids : List[Colloid]
        Returns
        Either all colloids in the vision cone if self.return_cone is True or sum of
        distances to colloids in vision cone if self.return_cone is False.
        """
        colls_in_vision = []
        colls_distance = []
        for other_p in other_colloids:
            dist = other_p.pos - my_pos
            dist_norm = jnp.linalg.norm(dist)
            in_range = dist_norm < self.vision_range
            if not in_range:
                continue
            in_cone = (
                jnp.arccos(jnp.dot(dist / dist_norm, my_director))
                < self.vision_half_angle
            )
            if in_cone and in_range:
                colls_in_vision.append(other_p)
                colls_distance.append(dist_norm)
        if self.return_cone is True:
            return colls_in_vision
        else:
            if len(colls_distance) != 0:
                summed_distance = jnp.sum(jnp.array(colls_distance))
                return jnp.array([summed_distance])
            else:
                return jnp.array([0.0])

    def compute_observable(self, colloid: Colloid, other_colloids: List[Colloid]):
        """
        Compute the vision cone of the colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.
        vision_angle : int
                Total angle of view in degrees
        Returns
        List of all particles in the colloids range of vision (return_cone = True) or
        sum of distances to each particle in range of vision (return_cone = False) or
        the normalised distance to the source if detect_source is true and the source is
        in the vision cone.
        """

        my_pos, my_director = self._calculate_director(colloid)

        # Check if source is in vision cone
        if self.detect_source:
            observable = self._detect_source(my_pos, my_director)

        # If detect source is off: calculate cone
        if not self.detect_source:
            observable = self._calculate_cone(my_pos, my_director, other_colloids)

        return observable
