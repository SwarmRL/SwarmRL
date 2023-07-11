"""
Module for the parent class of the tasks.

Notes
-----
The reward classes handle the computation of the reward from an environment and
compute the loss for the models to train on.
"""
from typing import List

import numpy as np

from swarmrl.models.interaction_model import Colloid


class SheepShepp:
    """
    Parent class for the reinforcement learning tasks.
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        center: np.ndarray = np.array([500, 500, 0]),
        radius=100,
        sheep_type: int = 1,
        shepp_type: int = 0,
    ):
        """
        Constructor for the reward class.

        Parameters
        ----------
        particle_type : int (default=0)
                Particle type to compute the reward for.
        """
        self.center = center
        self.radius = radius
        self.scale_factor = scale_factor
        self.sheep_type = sheep_type
        self.shepp_type = shepp_type
        self.old_positions = {"sheep_pos": [], "shepp_pos": []}

    def initialize(self, colloids: List[Colloid]):
        # populate the mask
        for colloid in colloids:
            if colloid.type == self.sheep_type:
                self.old_positions["sheep_pos"].append(colloid.pos)
            elif colloid.type == self.shepp_type:
                self.old_positions["shepp_pos"].append(colloid.pos)

        self.old_positions["sheep_pos"] = np.array(self.old_positions["sheep_pos"])
        self.old_positions["shepp_pos"] = np.array(self.old_positions["shepp_pos"])

    def __call__(self, colloids: List[Colloid]):
        sheep_pos = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.sheep_type]
        )
        shepps_pos = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.shepp_type]
        )

        old_dist = np.linalg.norm(
            self.old_positions["sheep_pos"][:, None, :]
            - self.old_positions["shepp_pos"][None, :, :],
            axis=-1,
        )

        # compute the distance between all sheeps and all shepps
        dists = np.linalg.norm(sheep_pos[:, None, :] - shepps_pos[None, :, :], axis=-1)

        delta = old_dist - dists
        reward = self.scale_factor * delta
        reward = np.where(reward >= 0, reward, 0)
        self.old_positions["sheep_pos"] = np.array(sheep_pos)
        self.old_positions["shepp_pos"] = np.array(shepps_pos)

        sheep_center_dist = np.linalg.norm(sheep_pos - self.center, axis=-1)
        reward2 = np.sum(np.where(sheep_center_dist < self.radius, 100, 0))
        total_reward = reward + reward2
        return total_reward
