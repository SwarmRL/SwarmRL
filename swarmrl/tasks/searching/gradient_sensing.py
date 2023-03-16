"""
Run and tumble task

This task uses the change in the gradient to determine whether a move was good or not.

Notes
-----
Requires a warm up step.
"""
from abc import ABC
from typing import List

import jax.numpy as np
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task


class GradientSensing(Task, ABC):
    """
    Find a location in a box using distances.
    """

    def __init__(
        self,
        source: np.ndarray = np.array([0, 0, 0]),
        decay_function: callable = None,
        box_length: np.ndarray = np.array([1.0, 1.0, 0.0]),
        reward_scale_factor: int = 10,
        particle_type: int = 0,
    ):
        """
        Constructor for the find origin task.

        Parameters
        ----------
        source : np.ndarray (default = (0, 0 0))
                Source of the gradient.
        decay_function : callable (required=True)
                A function that describes the decay of the field along one dimension.
                This cannot be left None. The function should take a distance from the
                source and return the magnitude of the field at this point.
        box_length : np.ndarray
                Side length of the box.
        reward_scale_factor : int (default=10)
                The amount the field is scaled by to get the reward.
        particle_type : int (default=0)

        """
        super().__init__(particle_type=particle_type)
        self.source = source / box_length
        self.decay_fn = decay_function
        self.reward_scale_factor = reward_scale_factor
        self.box_length = box_length

        # Class only attributes
        self._historic_positions = {}

    def initialize(self, colloids: List[Colloid]):
        """
        Prepare the task for running.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids to be used in the task.

        Returns
        -------
        observable :
                Returns the observable required for the task.
        """
        for item in colloids:
            if item.type == self.particle_type:
                index = onp.copy(item.id)
                position = onp.copy(item.pos) / self.box_length
                self._historic_positions[str(index)] = position

    def change_source(self, new_source: np.ndarray):
        """
        Changes the concentration field source.

        Parameters
        ----------
        new_source : np.ndarray
                Coordinates of the new source.
        """
        self.source = new_source

    def compute_colloid_reward(self, index: int, colloids):
        """
        Compute the reward for a single colloid.

        Parameters
        ----------
        index : int
                Index of the colloid to compute the reward for.

        Returns
        -------
        reward : float
                Reward for the colloid.
        """
        colloid_id = onp.copy(colloids[index].id)
        # Get the current position of the colloid
        current_position = onp.copy(colloids[index].pos) / self.box_length

        # Get the old position of the colloid
        old_position = self._historic_positions[str(colloid_id)]

        # Compute the distance from the source
        current_distance = np.linalg.norm(current_position - self.source)
        old_distance = np.linalg.norm(old_position - self.source)

        # Compute difference in scaled_distances
        delta = self.decay_fn(current_distance) - self.decay_fn(old_distance)

        # Compute the reward
        reward = np.clip(self.reward_scale_factor * delta, 0.0, None)

        # Update the historic position
        self._historic_positions[str(colloid_id)] = current_position

        return reward

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
        colloid_indices = self.get_colloid_indices(colloids)

        return [
            self.compute_colloid_reward(index, colloids) for index in colloid_indices
        ]
