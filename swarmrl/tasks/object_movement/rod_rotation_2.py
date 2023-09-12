"""
Class for rod rotation task.
"""
from typing import List

import jax.numpy as np
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task


class RotateRod2(Task):
    """
    Rotate a rod.
    """

    def __init__(
        self,
        rod_type: int = 1,
        particle_type: int = 0,
        reward_scale: int = 1,
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
        self.rod_type = rod_type
        self.reward_scale = reward_scale

        # Class only attributes
        self._historic_rod_director = None
        self.rod_index = None
        self.num_cols = 0

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
            if item.type == self.particle_type:
                self.num_cols += 1

        for item in colloids:
            if item.type == self.rod_type:
                self.rod_index = item.id
                self._historic_rod_director = onp.copy(item.director)
                break

    def _compute_reward(self, new_director: np.ndarray):
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

        angle = np.arctan2(
            np.cross(self._historic_rod_director[:2], new_director[:2]),
            np.dot(self._historic_rod_director[:2], new_director[:2]),
        )

        # Update the historical rod director
        self._historic_rod_director = new_director

        return self.reward_scale * np.abs(angle)

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
        rod_directors = colloids[self.rod_index].director

        reward = self._compute_reward(rod_directors)

        return reward * np.ones(self.num_cols)
