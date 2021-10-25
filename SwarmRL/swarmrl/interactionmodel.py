"""
Model to compute external forces in an espresso simulation.
"""
import torch
import numpy as np


class InteractionModel:
    """
    Parent class to compute external forces on colloids in an espresso simulation.
    """
    def compute_force(self, colloids: torch.Tensor) -> np.ndarray:
        """
        Compute the forces on a set of colloids.

        Parameters
        ----------
        colloids : tf.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        raise NotImplementedError
