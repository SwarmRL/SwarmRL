from abc import ABC
from typing import List

import jax.numpy as np

from swarmrl.models.interaction_model import Colloid
from swarmrl.tasks.task import Task


class FromGroup(Task, ABC):
    def __init__(
        self,
        particle_type: int = 0,
        box_length=np.array([1.0, 1.0, 0.0]),
        reward_scale_factor: int = 10,
    ):
        super().__init__(particle_type=particle_type)

        self.box_length = box_length
        self.old_pos = None
        self.particle_type = particle_type
        self.reward_scale_factor = reward_scale_factor
        self.indices = None

    def initialize(self, colloids: List[Colloid]):
        # get the indices of the colloids of the correct type
        self.indices = self.get_colloid_indices(colloids, p_type=self.particle_type)
        colloids = [colloids[i] for i in self.indices]

        # compute the positions of the colloids
        positions = np.array([col.pos for col in colloids]) / self.box_length
        # compute the center of mass of the positions
        center_of_mass = np.sum(positions, axis=0) / len(positions)
        # compute the distance between the center of mass and the colloids
        self.old_dists = np.linalg.norm(positions - center_of_mass, axis=-1)

    def __call__(self, colloids: List[Colloid]):
        colloids = [colloids[i] for i in self.indices]

        # compute the positions of the colloids
        positions = np.array([col.pos for col in colloids]) / self.box_length
        # compute the center of mass of the positions
        center_of_mass = np.sum(positions, axis=0) / len(positions)
        # compute the distance between the center of mass and the colloids
        current_distances = np.linalg.norm(positions - center_of_mass, axis=-1)
        # compute the difference between the current and historic distances
        diff_dist = self.old_dists - current_distances

        self.old_dists = current_distances

        # compute the reward
        rewards = self.reward_scale_factor * diff_dist
        return rewards
