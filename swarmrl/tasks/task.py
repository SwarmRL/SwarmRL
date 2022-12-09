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

    def __call__(self, observables: np.ndarray) -> float:
        """
        Compute the reward on the whole group of particles.

        Parameters
        ----------
        observables : np.ndarray (dimension, )
                Observable of a single colloid.

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
