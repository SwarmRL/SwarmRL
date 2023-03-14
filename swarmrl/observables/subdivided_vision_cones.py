"""
Computes vision cone(s).
"""
from typing import List

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable
from swarmrl.utils.utils import calc_signed_angle_between_directors


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
        radii: List[float],
        detected_types=None,
        particle_type: int = 0,
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
                if None then all will be detected.
        particle_type : int (default=0)
                Particle type to compute the observable for.
        """
        super().__init__(particle_type=particle_type)
        self.vision_range = vision_range
        self.vision_half_angle = vision_half_angle
        self.n_cones = n_cones
        self.radii = radii
        self.detected_types = detected_types
        self.angle_fn = jit(calc_signed_angle_between_directors)

    def _detect_all_things_to_see(self, colloids: List[Colloid]):
        """
        Determines what types of colloids are present in the simulation

        Parameters
        ----------
        colloids : object
                Colloids with all possible different types.

        Returns
        -------
        One dimensional np.array with all possible types
        in the corresponding index in which
        they will also later be found when the observable is calculated.
        """
        all_types = []
        for c in colloids:
            if c.type not in all_types:
                all_types.append(c.type)
        self.detected_types = np.array(np.sort(all_types))

    def _calculate_director(self, colloid: Colloid):
        """
        Calculates the normalised director of the colloids.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.

        Returns
        -------
        Colloid position and vision cone director.
        """
        my_pos = np.copy(colloid.pos)
        my_director = np.copy(colloid.director)
        # TODO: Add 3D support
        my_director = my_director / np.linalg.norm(my_director)
        assert abs(colloid.pos[2]) < 10e-6
        assert abs(colloid.director[2]) < 10e-6
        return my_pos, my_director

    def _calculate_cones_single_object(
        self,
        my_pos: np.ndarray,
        my_director: np.ndarray,
        c_type: int,
        c_pos: np.ndarray,
        radius: float,
    ):
        """
        Compute the vision cones of one colloid from one colloid.

        Parameters
        ----------
        my_pos : np.ndarray
                the 2D position of the colloid that has vision
        my_director : np.ndarray
                the 2D orientation of the colloid that has vision
        c_type : int
                The type of the colloid that is seen
        c_pos : np.ndarray
                The position of the colloid that is seen
        radius : float
                The radius of the colloid that has vision
        Returns
        -------
        np.ndarray of shape (n_cones, num_of_detected_types) containing
        the vision values for each cone and for each particle type that
        can be visible. At most one value is unequal to zero.
        """
        # generate a blue print of the output values
        vision_val_out = jnp.ones((self.n_cones, len(self.detected_types)))

        dist = c_pos - my_pos
        dist_norm = jnp.linalg.norm(dist)
        in_range = dist_norm < self.vision_range
        # check if output values will correspond with
        # no visible colloid at all because to far away
        vision_val_out *= in_range
        # calculation could stop here but is carried
        # on with zeros due to parallelization
        # keep on calculating even if there are only zeros

        # adjust the amplitude of the vision to the correct value
        vision_val_out *= jnp.min(jnp.array([1, 2 * radius / dist_norm]))

        # generate a mask that is only true for the 2D array entries with the right type
        type_mask = (
            jnp.ones((self.n_cones, len(self.detected_types)))
            * np.array(self.detected_types)[np.newaxis, :]
        )
        correct_type_mask = jnp.where(type_mask == c_type, True, False)
        # Apply the mask on the output values
        vision_val_out *= correct_type_mask

        # compare to the direction of view with
        # the direction in which the other colloid is.
        # Get the singed angle between them
        # call the jax.jit version of calc_signed_angle_between_directors()
        angle = self.angle_fn(my_director, dist / dist_norm)

        # get masks with True if the colloid is in the specific vision cone
        rims = (
            -self.vision_half_angle
            + jnp.arange(self.n_cones + 1) * self.vision_half_angle * 2 / self.n_cones
        )
        in_left_rim = jnp.where(rims[:-1] < angle, True, False)
        in_right_rim = jnp.where(rims[1:] > angle, True, False)
        in_a_cone = in_left_rim * in_right_rim
        # apply mask to the output_values
        vision_val_out *= in_a_cone[:, np.newaxis]

        # 2D array with right vision amplitude for
        # right type and cone only if distance < cut off radius
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
        other_colloids : List[Colloid]
                all the colloids besides the one with my_pos and my_director
        Returns
        -------
        np.array of shape (n_cones, num_of_types) containing the vision values
        for each cone and for each particle type that can be visible.
        """

        # prepare traceable colloid data for vectorized vision cone evaluation
        other_colloids_id = np.zeros((len(other_colloids)))
        other_colloids_pos = np.zeros((len(other_colloids), 3))
        other_colloids_types = np.zeros((len(other_colloids)))
        for index, c in enumerate(other_colloids):
            other_colloids_id[index] = c.id
            other_colloids_types[index] = c.type
            other_colloids_pos[index, :] = c.pos

        # wrap jax.vmap around the vision cone
        # evaluation concerning a single other colloid
        # therefore spare out a for loop over all colloid to be seen
        # make it parallelizable at the cost of calculating with sparse arrays
        calculate_cones = vmap(
            self._calculate_cones_single_object,
            in_axes=(None, None, 0, 0, 0),
            out_axes=0,
        )

        # executing the vectorized function
        vision_val_out_expanded = calculate_cones(
            my_pos,
            my_director,
            other_colloids_types,
            other_colloids_pos,
            np.array(self.radii),
        )
        # collapsing the data of every individual other_colloid and returning the result
        return np.sum(vision_val_out_expanded, axis=0)

    def compute_single_observable(
        self, index: int, colloids: List[Colloid]
    ) -> np.ndarray:
        """
        Compute the vision cones of the colloid.

        Parameters
        ----------
        index : int
                Index of colloid for which the observable should be computed.
        colloids
                colloids in the system.
        Returns
        -------
        np.array of shape (n_cones, num_of_detected_types) containing the vision values
        for each cone and for each particle type that can be visible.
        """
        colloid = colloids[index]

        if self.detected_types is None:
            self._detect_all_things_to_see(colloids)

        my_pos, my_director = self._calculate_director(colloid)

        of_others = [
            [c, self.radii[i]] for i, c in enumerate(colloids) if c is not index
        ]
        other_colloids = [of_others[i][0] for i in range(len(of_others))]
        self.radii = [of_others[i][1] for i in range(len(of_others))]

        observable = self._calculate_cones(my_pos, my_director, other_colloids)

        return observable

    def compute_observable(self, colloids: List[Colloid]):
        """
        Compute the vision cones of the colloids.

        Parameters
        ----------
        colloids
                colloids in the system.
        Returns
        -------
        np.array of shape (n_colloids, n_cones, num_of_detected_types)
        containing the vision values
        """
        reference_ids = self.get_colloid_indices(colloids)

        return [
            self.compute_single_observable(index, colloids) for index in reference_ids
        ]
