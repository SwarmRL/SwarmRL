"""
Class for rod rotation task.
"""

from typing import List

import jax
import jax.numpy as jnp
import numpy as onp

from swarmrl.components.colloid import Colloid
from swarmrl.tasks.task import Task
from swarmrl.utils.colloid_utils import compute_torque_partition_on_rod


class RodTorque(Task):
    """
    Push a Rod.
    """

    def __init__(
        self,
        rod_type: int = 1,
        particle_type: int = 0,
        direction: str = "CCW",
        angular_velocity_scale: int = 100,
        velocity_history_size: int = 100,
    ):
        """
        Constructor for the rod torque task.

        Parameters
        ----------
        rod_type : int (default=1)
                Type of particle making up the rod.
        particle_type : int (default=0)
                Type of particle receiving the reward.
        direction : str (default="CCW")
                Direction of the rod to rotate.
        angular_velocity_scale : float (default=100.0)
                The amount the velocity is scaled by to get the reward.
        velocity_history_size : int (default=100)
                Number of steps to average the velocity over.
        """
        super().__init__(particle_type=particle_type)
        self.rod_type = rod_type

        if velocity_history_size < 1:
            raise ValueError("Velocity history size must be greater than 0.")
        else:
            self.velocity_history_size = velocity_history_size

        if angular_velocity_scale < 1:
            raise ValueError(
                "Angular velocity scale must be greater than 0. For rotational"
                " direction, use 'CW' or 'CCW'."
            )

        if direction == "CW":
            angular_velocity_scale *= -1  # CW is negative

        self.angular_velocity_scale = angular_velocity_scale
        self._velocity_history_list = jnp.full(velocity_history_size, jnp.nan)

        self.decomp_fn = jax.jit(compute_torque_partition_on_rod)

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
        self._velocity_history_list = jnp.full(self.velocity_history_size, jnp.nan)
        for item in colloids:
            if item.type == self.rod_type:
                self._historic_rod_director = onp.copy(item.director)
                break

    def _compute_torque_on_rod(
        self,
        rod_positions: jnp.ndarray,
        colloid_directors: jnp.ndarray,
        colloid_positions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the torques on the rod.

        Parameters
        ----------
        rod_positions : jnp.ndarray (n_rod, 3)
                Positions of the rod particles.
        colloid_directors : jnp.ndarray (n_colloids, 3)
                Directors of the colloids.
        colloid_positions : jnp.ndarray (n_colloids, 3)
                Positions of the colloids.

        Returns
        -------
        torques : jnp.ndarray (n_colloids, )
                Torques on the rod for each colloid.
        """
        torques = self.decomp_fn(rod_positions, colloid_directors, colloid_positions)[
            :, 2
        ]
        return torques

    def _compute_angular_velocity(self, new_director: jnp.ndarray):
        """
        Compute the average angular velocity of the rod. This gets clipped,
        so that negative values become 0.

        Parameters
        ----------
        new_director : jnp.ndarray (3, )
                New rod director.

        Returns
        -------
        angular_velocity : float
                Angular velocity of the rod
        """
        angular_velocity = jnp.arctan2(
            jnp.cross(self._historic_rod_director[:2], new_director[:2]),
            jnp.dot(self._historic_rod_director[:2], new_director[:2]),
        )

        # Convert to degree for easier handling
        angular_velocity = jnp.rad2deg(angular_velocity)

        # Update the historical rod director and velocity.
        self._historic_rod_director = new_director
        self._velocity_history_list = jnp.roll(self._velocity_history_list, -1)
        self._velocity_history_list = self._velocity_history_list.at[-1].set(
            angular_velocity
        )

        # Return the clipped average velocity. nanmean ignores the not-yet-filled
        # entries so the mean is unbiased before the history buffer is full.
        if self.angular_velocity_scale > 0.0:
            return jnp.clip(jnp.nanmean(self._velocity_history_list), 0.0, None)
        else:
            return jnp.clip(jnp.nanmean(self._velocity_history_list), None, 0.0)

    def sort_out_opposite_direction_torques(
        self,
        colloid_torques_on_rod: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Remove rewards for torque in the wrong direction.

        Parameters
        ----------
        colloid_torque_on_rod : jnp.ndarray (n_colloids, )
                Torques of the colloids on the rod.

        Returns
        -------
        torques_in_direction : jnp.ndarray (n_colloids, )
                Torques on the rod for each colloid with wrong directions set to 0.
        """
        # Sign of the torques has to be inverted to avoid negative rewards.
        sign = jnp.sign(self.angular_velocity_scale)
        return jnp.clip(-sign * colloid_torques_on_rod, 0.0, None) * sign

    def _compute_torque_and_velocity_reward(
        self,
        rod_directors: jnp.ndarray,
        rod_positions: jnp.ndarray,
        colloid_directors: jnp.ndarray,
        colloid_positions: jnp.ndarray,
    ):
        """
        Get the torques, the turning velocity and apply the scaling.

        Note that only torques into the desired direction are considered for
        reward computation; torques in the opposite direction are clipped to 0.

        Parameters
        ----------
        rod_directors : jnp.ndarray (n_rod, 3)
                Directors of the rod.
        rod_positions : jnp.ndarray (n_rod, 3)
                Positions of the rod particles.
        colloid_directors : jnp.ndarray (n_colloids, 3)
                Directors of the colloids.
        colloid_positions : jnp.ndarray (n_rod, 3)
                Positions of the colloids.

        Returns
        -------
        rewards : jnp.ndarray (n_colloids, )
                Rewards for each colloid.
        """

        colloid_torques_on_rod = self._compute_torque_on_rod(
            rod_positions, colloid_directors, colloid_positions
        )
        torques = self.sort_out_opposite_direction_torques(colloid_torques_on_rod)
        velocity = self._compute_angular_velocity(rod_directors[0])

        return torques * velocity * self.angular_velocity_scale

    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward.

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
        rod_data = [(c.pos, c.director) for c in colloids if c.type == self.rod_type]
        rod_positions, rod_directors = map(jnp.array, zip(*rod_data))

        colloid_data = [
            (c.pos, c.director) for c in colloids if c.type == self.particle_type
        ]
        colloid_positions, colloid_directors = map(jnp.array, zip(*colloid_data))

        rewards = self._compute_torque_and_velocity_reward(
            rod_directors, rod_positions, colloid_directors, colloid_positions
        )

        return rewards
