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
        center: np.ndarray = np.array([500, 500, 0]),
        radius=150,
        fear_radius=100,
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
        self.fear_radius = fear_radius
        self.sheep_type = sheep_type
        self.shepp_type = shepp_type
        self.colls = {"sheeps": 0, "shepps": 0}
        self.sheep_mask = None
        self.shepp_mask = None

    def initialize(self, colloids: List[Colloid]):
        # populate the mask
        for colloid in colloids:
            if colloid.type == self.sheep_type:
                self.colls["sheeps"] += 1
            elif colloid.type == self.shepp_type:
                self.colls["shepps"] += 1

    def __call__(self, colloids: List[Colloid]):
        sheeps = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.sheep_type]
        )
        shepps = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.shepp_type]
        )

        # compute the distance between all sheeps and shepps
        dist_sheep_shepps = np.linalg.norm(
            sheeps[:, None, :] - shepps[None, :, :], axis=2
        )
        dist_reward = np.where(
            np.any(dist_sheep_shepps > self.fear_radius, axis=1), 0, 1
        )
        # compute the distance between sheeps and center
        dist_sheeps = np.linalg.norm(sheeps - self.center, axis=1)
        sheep_reward = np.where(dist_sheeps > self.radius, 1, -1)

        shepp_reward = np.ones(self.colls["shepps"]) * int(
            not np.any(dist_sheeps > self.radius, axis=0)
        )
        # the the two rewards together
        sheep_reward = sheep_reward + dist_reward
        rewards = {
            str(self.sheep_type): sheep_reward,
            str(self.shepp_type): shepp_reward,
        }

        return rewards
