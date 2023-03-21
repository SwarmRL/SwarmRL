"""
Class for multi-tasking.
"""
from swarmrl.tasks.task import Task
from typing import List
from swarmrl.models.interaction_model import Colloid
import jax.numpy as np


class MultiTasking(Task):
    """
    Class for handling multiple tasks.
    """
    def __init__(self, particle_type: int = 0, tasks: List[Task] = []):
        """
        Constructor for multi-tasking.
        """
        super().__init__(particle_type)
        self.tasks = tasks

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observables as needed.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Some of the observables passed to the constructor might need to be
        initialized with the positions of the colloids. This method does
        that.
        """
        for item in self.tasks:
            item.initialize(colloids)

    def __call__(self, colloids: List[Colloid]) -> np.ndarray:
        """
        Computes all observables and returns them in a concatenated list.

        Parameters
        ----------
        colloids : list of all colloids.

        Returns
        -------
        rewards : np.ndarray of shape (num_colloids, )
                Array of rewards for each colloid.
        """
        species_indices = self.get_colloid_indices(colloids)
        rewards = np.zeros(len(species_indices))
        for task in self.tasks:
            ts = task(colloids)
            rewards += ts

        return rewards
