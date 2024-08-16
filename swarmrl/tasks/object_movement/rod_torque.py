"""
Class for rod rotation task.
"""

from typing import List

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.components.colloid import Colloid
from swarmrl.tasks.task import Task
from swarmrl.utils.colloid_utils import compute_torque_on_rod


class RodTorque(Task):
    """
    Put Torque on a Rod.
    """

    def __init__(
        self,
        rod_type: int = 1,
        particle_type: int = 0,
        direction: str = "CCW",
        angular_velocity_scale: int = 1,
    ):
        """
        Constructor for the find origin task.

        Parameters
        ----------
        partition : bool (default=True)
                Whether to partition the reward by particle contribution.
        rod_type : int (default=1)
                Type of particle making up the rod.
        scale_factor : float (default=100.0)
                The amount the velocity is scaled by to get the reward.
        direction : Union[None, str] (default=None)
                Direction of the rod to rotate. If None, the rod will
                rotate arbitrarily.
        particle_type : int (default=0)
                Type of particle receiving the reward.
        velocity_history : int (default=100)
                Number of steps to average the velocity over.
        """
        super().__init__(particle_type=particle_type)
        self.rod_type = rod_type

        if direction == "CW":
            angular_velocity_scale *= -1  # CW is negative

        self.angular_velocity_scale = angular_velocity_scale

        # Class only attributes
        self._historic_rod_director = None

        self.decomp_fn = jax.jit(compute_torque_on_rod)

    def initialize(self, colloids: List[Colloid]):
        """
        Prepare the task for running.

        In this case, as all rod directors are the same, we
        only need to take on for the historical value.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids to be used in the task.

        Returns
        -------
        Updates the class state.
        """
        for item in colloids:
            if item.type == self.rod_type:
                self._historic_rod_director = onp.copy(item.director)
                break

    def _torque_on_rod(
        self,
        colloid_positions: np.ndarray,
        rod_positions: np.ndarray,
        rod_directors: np.ndarray,
    )-> np.ndarray:
        """
        Compute the torques on the rod.

        Parameters
        ----------
        colloid_positions : np.ndarray (n_colloids, 3)
                Positions of the colloids.
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        rod_directors : np.ndarray (n_rod, 3)
                Directors of the rod particles.

        Returns
        -------
        torques : np.ndarray (n_colloids, )
                Torques on the rod for each colloid.
        """
        torques = self.decomp_fn(colloid_positions, rod_positions, rod_directors)[:,2]
        return torques

    def _torque_partition(
        self,
        colloids_torques_on_rod: np.ndarray,
        )-> np.ndarray:                                       # Negative Torques entsprechen CCW-Rotation Also muss das Vorzeichen fuer Belohnungen einmal umgedreht werden
        """
        Remove rewards for torque in the wrong direction and apply the scaling.

        Parameters
        ----------
        colloid_torques_on_rod : np.ndarray (n_colloids, )
                Torques of the colloids on the rod.

        Returns
        -------
        torques_in_direction : np.ndarray (n_colloids, )
                Torques on the rod for each colloid with wrong directions set to 0.
        """
        if self.angular_velocity_scale > 0.0:
            torques_in_direction = colloids_torques_on_rod.at[colloids_torques_on_rod > 0.0].set(0.0)
        else:
            torques_in_direction = colloids_torques_on_rod.at[colloids_torques_on_rod < 0.0].set(0.0)

        return torques_in_direction * -1 * self.angular_velocity_scale

    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward.

        In this case of this task, the observable itself is the gradient of the field
        that the colloid is swimming in. Therefore, the change is simply scaled and
        returned.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of colloids to be used in the task.

        Returns
        -------
        rewards : List[float] (n_colloids, )
                Rewards for each colloid.
        """
        # Collect the important data.
        rod = [colloid for colloid in colloids if colloid.type == self.rod_type]
        rod_positions = np.array([colloid.pos for colloid in rod])
        rod_directors = np.array([colloid.director for colloid in rod])

        chosen_colloids = [
            colloid for colloid in colloids if colloid.type == self.particle_type
        ]
        colloid_positions = np.array([colloid.pos for colloid in chosen_colloids])

        colloids_torque_on_rod = self._torque_on_rod(colloid_positions, rod_positions, rod_directors)

        torques_in_turning_direction = self._torque_partition(colloids_torque_on_rod)

        return torques_in_turning_direction
