"""
Module for the parent class of the tasks.

Notes
-----
The reward classes handle the computation of the reward from an environment and
compute the loss for the models to train on.
"""
import jax.numpy as np


class Task:
    """
    Parent class for the reinforcement learning tasks.
    """

    def __init__(self):
        """
        Constructor for the reward class.
        """
        pass

    def __call__(
        self,
        observables: np.ndarray,
        colloid: object,
        colloids: list,
        other_colloids: list,
    ) -> float:
        """
        Compute the reward on the whole group of particles.

        Parameters
        ----------
        observable : np.ndarray (dimension, )
                Observable of a single colloid.
        colloid : object
                The colloid, the reward should be computed for.
        colloids : list
                A list of all colloids in the system.
        other_colloids : list
                A list of all other colloids in the system.
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
