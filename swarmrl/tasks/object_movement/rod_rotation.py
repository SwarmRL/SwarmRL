"""
Class for rod rotation task.
"""
from typing import List

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.models.interaction_model import Colloid
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
        scale_factor: float = 100.0,
        particle_type: int = 0,
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
        particle_type : int (default=0)
                Type of particle receiving the reward.
        """
        super().__init__(particle_type=particle_type)
        self.partition = partition
        self.rod_type = rod_type
        self.scale_factor = scale_factor

        # Class only attributes
        self._historic_rod_director = None

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
        # Compute the angular velocity
        angular_velocity = np.arccos(np.dot(self._historic_rod_director, new_director))

        # Update the historical rod director
        self._historic_rod_director = new_director

        return angular_velocity

    def partition_reward(
        self,
        reward: float,
        colloid_positions: np.ndarray,
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
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        rod_directors : np.ndarray (n_rod, 3)
                Directors of the rod particles.

        Returns
        -------
        partitioned_reward : np.ndarray (n_colloids, )
                Partitioned reward for each colloid.
        """
        colloid_partitions = self.decomp_fn(
            colloid_positions, rod_positions, rod_directors
        )
        return reward * colloid_partitions

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

        # Compute angular velocity
        angular_velocity = self.scale_factor * self._compute_angular_velocity(
            rod_directors[0]
        )
        reward = angular_velocity

        # Compute colloid-wise rewards
        return self.partition_reward(
            reward, colloid_positions, rod_positions, rod_directors
        )
