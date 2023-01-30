"""
Computes vision cone(s).
"""
from abc import ABC
from typing import List

import numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task


class SubdividedVisionCones(Observable, ABC, Task):
    """
    The vision cone acts like a camera for the particles. It can either output all
    colloids within its view, a function of the distances between one colloid and all
    other colloids, or the normalised distance to the source.
    """

    def __init__(
        self, vision_range: float, vision_half_angle: float, n_cones: int, radii=[]
    ):
        """
        Constructor for the observable.

        Parameters
        ----------
        vision_range : float
                How far the particles can see.
        vision_half_angle : float
                the half width of the field of view in radiant units
        n_cones: int
                In how many cone is the field of view subdivided
        radii: list
                List of the radii of the colloids in the experiment,
                ordered by the ids of the colloids
        """
        self.vision_range = vision_range
        self.vision_half_angle = vision_half_angle
        self.n_cones = n_cones
        self.radii = radii
        self.types = []

    def initialize(self, colloids: List[Colloid]):
        """
        This method is not needed for the vision_cone because the initialization
        currently only initialises general features for all colloids while the vision
        cone needs to be calculated for each cone separately.
        """
        pass

    def detect_things_to_see(self, colloids: List[Colloid]):
        """
        Calculates the normalised director of the colloids.

        Parameters
        ----------
        colloids : object
                Colloids with all possible different types.

        Returns
        all possible types in the corresponding index they will also later be found in,
        when the observable is calculated
        """
        types = []
        for c in colloids:
            if c.type not in types:
                types.append(c.type)
        self.types = np.array(types)

    def _calculate_director(self, colloid):
        """
        Calculates the normalised director of the colloids.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.

        Returns
        Colloid position and vision cone director.
        """
        my_pos = np.copy(colloid.pos[:2])
        my_director = np.copy(colloid.director[:2])
        # TODO: Add 3D support
        my_director = my_director / np.linalg.norm(my_director)
        return my_pos, my_director

    def _calculate_cones(self, my_pos, my_director, colloids: List[Colloid]):
        """
        Calculates the vision cones of the colloid.

        Parameters
        ----------
        my_pos : np.ndarray
                Position of the colloid.
        my_director : np.ndarray
                Normalised director of the colloid.
        colloids : List[Colloid]
                all colloids
        Returns
        np.array of shape (n_cones, num_of_types) containing the vision values
        for each cone and for each particle type that can be visible.
        """
        vision_val_out = np.zeros((self.n_cones, len(self.types)))

        for c in colloids:
            dist = c.pos[:2] - my_pos
            dist_norm = np.linalg.norm(dist)
            in_range = dist_norm < self.vision_range
            # make sure not to see yourself ;
            # put vision_range sufficiently high if no upper limit is wished
            if dist_norm != 0 and not in_range:
                continue
            # calc peceived angle deviation ( sign of angle is missing )
            angle = np.arccos(np.dot(dist / dist_norm, my_director))
            # use the director in orthogonal direction to determine sign
            orthogonal_dot = np.dot(dist / dist_norm, [-my_director[1], my_director[0]])
            angle *= np.sign(orthogonal_dot)
            for cone in range(self.n_cones):
                in_left_rim = (
                    -self.vision_half_angle
                    + cone * self.vision_half_angle * 2 / self.n_cones
                    < angle
                )
                in_right_rim = (
                    angle
                    < -self.vision_half_angle
                    + (cone + 1) * self.vision_half_angle * 2 / self.n_cones
                )
                # sort the perceived colloid c by their vision cone and type
                if in_left_rim and in_right_rim:
                    type_num = np.where(self.types == c.type)[0][0]
                    vision_val_out[cone, type_num] += np.min(
                        [1, 2 * self.radii[c.id] / dist_norm]
                    )
            return vision_val_out

    def compute_observable(self, colloid: Colloid, colloids: List[Colloid]):
        """
        Compute the vision cones of the colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        colloids
                colloids in the system.
        vision_angle : int
                Total angle of view in degrees
        Returns
        np.array of shape (n_cones, num_of_types) containing the vision values
        for each cone and for each particle type that can be visible.
        """
        if self.types == []:
            self.detect_things_to_see(colloids)

        my_pos, my_director = self._calculate_director(colloid)

        observable = self._calculate_cones(my_pos, my_director, colloids)

        return observable
