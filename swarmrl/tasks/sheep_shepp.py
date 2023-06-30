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
        self.sheep_type = sheep_type
        self.shepp_type = shepp_type
        self.colls = {"sheeps": 0, "shepps": 0}
        self.old_dists = None

    def initialize(self, colloids: List[Colloid]):
        # populate the mask
        for colloid in colloids:
            if colloid.type == self.sheep_type:
                self.colls["sheeps"] += 1
            elif colloid.type == self.shepp_type:
                self.colls["shepps"] += 1

        sheeps = [c for c in colloids if c.type == self.sheep_type]
        predators = [p for p in colloids if p.type == self.shepp_type]
        sheep_pos = np.array([s.pos for s in sheeps])
        pred_pos = np.array([p.pos for p in predators])

        # distances between sheeps and predators
        dr_p = pred_pos[:, None, :] - sheep_pos[None, :, :]
        dr_norm_p = np.linalg.norm(dr_p, axis=-1)
        self.old_dists = dr_norm_p

    def __call__(self, colloids: List[Colloid]):
        sheeps = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.sheep_type]
        )
        shepps = np.array(
            [colloid.pos for colloid in colloids if colloid.type == self.shepp_type]
        )

        dist_sheeps = np.linalg.norm(sheeps - self.center, axis=1)

        # compute the reward for the sheeps
        n_sheeps = np.sum(dist_sheeps < self.radius)

        # compute the reward for going to the sheeps
        dr_p = shepps[:, None, :] - sheeps[None, :, :]
        dr_norm_p = np.linalg.norm(dr_p, axis=-1)

        r = np.sum(
            np.where(dr_norm_p < self.old_dists, self.old_dists - dr_norm_p, 0), axis=0
        )

        self.old_dists = dr_norm_p

        shepp_reward = 100 * np.ones(self.colls["shepps"]) * n_sheeps

        # the the two rewards together
        # sheep_reward = sheep_reward + dist_reward
        rewards = {str(self.sheep_type): [], str(self.shepp_type): shepp_reward + r}

        return rewards
