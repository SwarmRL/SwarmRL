from abc import ABC
from typing import List

import jax.numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task


class FromGroup(Task, ABC):
    def __init__(
        self,
        particle_type: int = 0,
        box_length: np.ndarray = np.array([1.0, 1.0, 0.0]),
        reward_scale_factor: int = 10,
    ):
        super().__init__(particle_type=particle_type)

        self.box_length = box_length
        self.historic_distances = {}
        self.particle_type = particle_type
        self.reward_scale_factor = reward_scale_factor

    def initialize(self, colloids: List[Colloid]):
        # get the indices of the colloids of the correct type
        indices = self.get_colloid_indices(colloids, p_type=self.particle_type)
        colloids = [colloids[i] for i in indices]

        # compute the positions of the colloids
        positions = np.array([col.pos for col in colloids]) / self.box_length
        # compute the center of mass of the positions
        center_of_mass = np.sum(positions, axis=0) / len(positions)
        # compute the distance between the center of mass and the colloids
        distances = np.linalg.norm(positions - center_of_mass, axis=-1)
        self.historic_distances = {
            col.id: distances[i] for i, col in enumerate(colloids)
        }

    def __call__(self, colloids: List[Colloid]):
        indices = self.get_colloid_indices(colloids, p_type=self.particle_type)
        colloids = [colloids[i] for i in indices]

        # compute the positions of the colloids
        positions = np.array([col.pos for col in colloids]) / self.box_length
        # compute the center of mass of the positions
        center_of_mass = np.sum(positions, axis=0) / len(positions)
        # compute the distance between the center of mass and the colloids
        current_distances = np.linalg.norm(positions - center_of_mass, axis=-1)
        # compute the difference between the current and historic distances
        diff_dist = []
        for i, col in enumerate(colloids):
            diff_dist.append(current_distances[i] - self.historic_distances[col.id])
            # update the historic distances
            self.historic_distances[col.id] = current_distances[i]
        # compute the reward
        return -1 * self.reward_scale_factor * np.array(diff_dist)
