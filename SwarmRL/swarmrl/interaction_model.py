"""
Model to compute external forces in an espresso simulation.
"""
import torch
import numpy as np


class InteractionModel(torch.nn.Module):
    """
    Parent class to compute external forces on colloids in an espresso simulation.

    Inherits from the module class of Torch. When the class is called, the forward
    method is run.
    """
    def compute_force(self, colloids: torch.Tensor) -> np.ndarray:
        """
        Compute the forces on a set of colloids.

        Parameters
        ----------
        colloids : torch.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        raise NotImplementedError("Implemented in child classes.")

    def forward(self, colloids: torch.Tensor, state: torch.Tensor = None):
        """
        Perform the forward pass over the model.

        In this method, all other stages of the model should be called. In the case of
        a simple algebraic model, this should just be a call to compute_force.

        Parameters
        ----------
        colloids : torch.Tensor
                Tensor of colloids on which to operate. shape=(n_colloids, n_properties)
                where properties can very between test_models.
        state : torch.Tensor
                State of the system on which a reward may be computed. Defaults to None
                to allow for non-NN models.

        Returns
        -------
        forces : np.ndarray
                Numpy array of forces to apply to the colloids. shape=(n_colloids, 3)
        """
        raise NotImplementedError("Implemented in the child classes.")
