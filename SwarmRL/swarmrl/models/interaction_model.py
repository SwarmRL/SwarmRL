"""
Model to compute external forces in an espresso simulation.
"""
import torch
import numpy as np


class InteractionModel(torch.nn.Module):
    """
    Parent class to compute external forces on colloids in a simulation.
    Inherits from the module class of Torch.
    """
    def calc_force(self, colloid, other_colloids) -> np.ndarray:
        """
        Calculate the forces that will be applied to ``colloid``
        Parameters
        ----------
        colloid: object with a ``pos``, ``v``, ``mass`` and ``director`` attribute
        other_colloids: list of colloids

        Returns
        -------
        np.array of three floats: the force
        """
        raise NotImplementedError("Implemented in child classes.")

    def calc_torque(self, colloid, other_colloids) -> np.ndarray:
        """
        See ``calc_force``
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
