"""
Computes vision cone(s).
"""
import sys
sys.path.append('swarmrl/observables')
sys.path.append('swarmrl/models')

from typing import List

import jax.numpy as jnp
import numpy as np
from jax import vmap

from interaction_model import Colloid
from observable import Observable


class SubdividedVisionCones(Observable):
    """
    The vision cone acts like a camera for the particles. It can either output all
    colloids within its view, a function of the distances between one colloid and all
    other colloids, or the normalised distance to the source.
    """

    def __init__(
        self,
        vision_range: float,
        vision_half_angle: float,
        n_cones: int,
        radii=[],
        detected_types=None,
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
        detected_types: list
                list of colloid types to be detected in requested order.
                For example [0,2] here colloids of type 1 won't be detected.
                if None then al will be detected.

        """
        self.vision_range = vision_range
        self.vision_half_angle = vision_half_angle
        self.n_cones = n_cones
        self.radii = radii
        self.detected_types = detected_types

    def initialize(self, colloids: List[Colloid]):
        """
        This method is not needed for the vision_cone because the initialization
        currently only initialises general features for all colloids while the vision
        cone needs to be calculated for each cone separately.
        """
        pass

    def detect_all_things_to_see(self, colloids: List[Colloid]):
        """
        Determines what types of colloids are present in the simulation

        Parameters
        ----------
        colloids : object
                Colloids with all possible different types.

        Returns
        One dimensional np.array with all possible types
        in the corresponding index in which
        they will also later be found when the observable is calculated.
        """
        all_types = []
        for c in colloids:
            if c.type not in all_types:
                all_types.append(c.type)
        self.detected_types = np.array(np.sort(all_types))

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

    def _calculate_cones_single_object(
        self, my_pos, my_director,c_id,c_type,c_pos,radius
    ):

        vision_val_out = jnp.ones((self.n_cones, len(self.detected_types)))
        type_mask = jnp.ones((self.n_cones, len(self.detected_types))) * np.array(self.detected_types)[np.newaxis,:]
        correct_type_mask=jnp.where(type_mask==c_type,True,False)
        vision_val_out*=correct_type_mask

        #should not be necessary
        #correct_type = c_type in self.detected_types
        #vision_val_out*=correct_type

        dist = c_pos - my_pos
        dist_norm = jnp.linalg.norm(dist)
        in_range = dist_norm < self.vision_range
        vision_val_out*=in_range

        vision_val_out*=jnp.min(jnp.array([1, 2 * radius / dist_norm]))

        angle = jnp.arccos(jnp.dot(dist / dist_norm, my_director))
        # use the director in orthogonal direction to determine sign
        orthogonal_dot = jnp.dot(dist / dist_norm, jnp.array([-my_director[1], my_director[0]]))
        angle *= jnp.sign(orthogonal_dot)

        rims=-self.vision_half_angle + jnp.arange(self.n_cones+1) * self.vision_half_angle * 2 / self.n_cones
        in_left_rim=jnp.where(rims[:-1]<angle,True,False)
        in_right_rim=jnp.where(rims[1:]>angle,True,False)
        in_a_cone=in_left_rim*in_right_rim
        vision_val_out*=in_a_cone[:,np.newaxis]

        return vision_val_out

    def _calculate_cones(self, my_pos, my_director, other_colloids: List[Colloid]):
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
        # vision_val_out_expanded =
        # np.zeros((self.n_cones, len(self.detected_types), len(colloids)))
        other_colloids_id=np.zeros((len(other_colloids)))
        other_colloids_pos=np.zeros((len(other_colloids),2))
        other_colloids_types=np.zeros((len(other_colloids)))
        for index,c in enumerate(other_colloids):
            other_colloids_id[index]=c.id
            other_colloids_types[index]=c.type
            other_colloids_pos[index,:]=c.pos[:2]

        calculate_cones = vmap(
            self._calculate_cones_single_object,
            in_axes=(None, None,0,0,0,0),
            out_axes=0,
        )
        vision_val_out_expanded = calculate_cones(
            my_pos, my_director,other_colloids_id, other_colloids_types, other_colloids_pos, np.array(self.radii)
        )
        return np.sum(vision_val_out_expanded, axis=0)

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
        np.array of shape (n_cones, num_of_detected_types) containing the vision values
        for each cone and for each particle type that can be visible.
        """
        if self.detected_types is None:
            self.detect_all_things_to_see(colloids)

        my_pos, my_director = self._calculate_director(colloid)

        of_others=[[c,self.radii[i]] for i,c in enumerate(colloids) if c is not colloid]
        other_colloids = [of_others[i][0] for i in range(len(of_others))]
        self.radii = [of_others[i][1] for i in range(len(of_others))]

        observable = self._calculate_cones(my_pos, my_director, other_colloids)

        return observable
