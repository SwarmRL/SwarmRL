"""
Class for rod rotation task.
"""

from typing import List

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.components.colloid import Colloid
from swarmrl.tasks.task import Task
from swarmrl.utils.colloid_utils import compute_torque_partition_on_rod


class RotateRod(Task):
    """
    Rotate a rod.
    """

    def __init__(
        self,
        partition: bool = True,
        rod_type: int = 1,
        particle_type: int = 0,
        direction: str = "CCW",
        angular_velocity_scale: int = 1,
        velocity_history: int = 100,
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
        self.partition = partition
        self.rod_type = rod_type
        self.velocity_history = velocity_history

        if direction == "CW":
            angular_velocity_scale *= -1  # CW is negative

        self.angular_velocity_scale = angular_velocity_scale
        self._velocity_history = np.zeros(velocity_history)
        self._append_index = int(velocity_history - 1)

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
        self._velocity_history = np.zeros(self.velocity_history)
        self._append_index = int(self.velocity_history - 1)
        for item in colloids:
            if item.type == self.rod_type:
                self._historic_rod_director = onp.copy(item.director)
                break

    def _compute_angular_velocity(self, new_director: np.ndarray):
        """
        Compute the instantaneous angular velocity of the rod.

        Parameters
        ----------
        new_director : np.ndarray (3, )
                New rod director.

        Returns
        -------
        angular_velocity : float
                Angular velocity of the rod
        """
        angular_velocity = np.arctan2(
            np.cross(self._historic_rod_director[:2], new_director[:2]),
            np.dot(self._historic_rod_director[:2], new_director[:2]),
        )

        # Convert to rph for better scaling
        angular_velocity = ((np.rad2deg(angular_velocity) / 10.0) / 360) * 60 * 60

        # Update the historical rod director and velocity.
        self._historic_rod_director = new_director
        self._velocity_history = np.roll(self._velocity_history, -1)
        self._velocity_history = self._velocity_history.at[self._append_index].set(
            angular_velocity
        )

        # Return the scaled average velocity.
        return np.clip(
            self.angular_velocity_scale * np.nanmean(self._velocity_history), 0.0, None
        )

    def partition_reward(
        self,
        reward: float,
        colloid_positions: np.ndarray,
        colloid_directors: np.ndarray,
        rod_positions: np.ndarray,
        rod_directors: np.ndarray,
    ) -> np.ndarray:
        """
        Partition a reward into colloid contributions.

        Parameters
        ----------
        reward : float
                Reward to be partitioned.
        colloid_positions : np.ndarray (n_colloids, 3)
                Positions of the colloids.
        colloid_directors : np.ndarray (n_colloids, 3)
                Directors of the colloids.
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        rod_directors : np.ndarray (n_rod, 3)
                Directors of the rod particles.

        Returns
        -------
        partitioned_reward : np.ndarray (n_colloids, )
                Partitioned reward for each colloid.
        """
        if self.partition:
            colloid_partitions = self.decomp_fn(
                colloid_positions, colloid_directors, rod_positions, rod_directors
            )
        else:
            colloid_partitions = (
                np.ones(colloid_positions.shape[0]) / colloid_positions.shape[0]
            )

        return reward * colloid_partitions

    def _compute_angular_velocity_reward(
        self,
        rod_directors: np.ndarray,
        rod_positions: np.ndarray,
        colloid_positions: np.ndarray,
        colloid_directors: np.ndarray,
    ):
        """
        Compute the angular velocity reward.

        Parameters
        ----------
        rod_directors : np.ndarray (n_rod, 3)
                Directors of the rod particles.
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        colloid_positions : np.ndarray (n_colloids, 3)
                Positions of the colloids.
        colloid_directors : np.ndarray (n_colloids, 3)
                Directors of the colloids.

        Returns
        -------
        angular_velocity_reward : float
                Angular velocity reward.
        """
        # Compute angular velocity
        angular_velocity = self._compute_angular_velocity(rod_directors[0])
        # Compute colloid-wise rewards
        return self.partition_reward(
            angular_velocity,
            colloid_positions,
            colloid_directors,
            rod_positions,
            rod_directors,
        )

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
        colloid_directors = np.array([colloid.director for colloid in chosen_colloids])

        # Compute the angular velocity reward
        angular_velocity_term = self._compute_angular_velocity_reward(
            rod_directors, rod_positions, colloid_positions, colloid_directors
        )

        return angular_velocity_term
