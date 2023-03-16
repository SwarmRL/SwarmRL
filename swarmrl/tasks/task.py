"""
Module for the parent class of the tasks.

Notes
-----
The reward classes handle the computation of the reward from an environment and
compute the loss for the models to train on.
"""
from typing import List

from swarmrl.models.interaction_model import Colloid


class Task:
    """
    Parent class for the reinforcement learning tasks.
    """

    def __init__(self, particle_type: int = 0):
        """
        Constructor for the reward class.

        Parameters
        ----------
        particle_type : int (default=0)
                Particle type to compute the reward for.
        """
        self.particle_type = particle_type

    def get_colloid_indices(self, colloids: List[Colloid], p_type: int = None):
        """
        Get the indices of the colloids in the observable of a specific type.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids from which to get the indices.
        p_type : int (default=None)
                Type of the colloids to get the indices for. If None, the
                particle_type attribute of the class is used.


        Returns
        -------
        indices : List[int]
                List of indices for the colloids of a particular type.
        """
        if p_type is None:
            p_type = self.particle_type

        indices = []
        for i, colloid in enumerate(colloids):
            if colloid.type == p_type:
                indices.append(i)

        return indices

    def __call__(self, colloids: List[Colloid]) -> float:
        """
        Compute the reward on the whole group of particles.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, dimension)
                List of colloid objects in the system.

        Returns
        -------
        Reward : float
                Reward for the current state.

        Examples
        --------
        my_task = Task()
        reward = my_task(state)
        """
        raise NotImplementedError("Implemented in child class.")
